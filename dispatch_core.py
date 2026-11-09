"""Core data pipeline and route-insertion costs for the Meituan dispatch project.

Everything here is plain Python + numpy + pandas so it runs in Colab. The Pyomo models live in dispatch_models.py.
"""
import ast
import itertools
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------- loading


def load_tables(data_dir):
    """Read the four challenge tables from data_dir (the folder that holds the CSV files)."""
    import os, zipfile
    wb_csv = os.path.join(data_dir, "all_waybill_info_meituan_0322.csv")
    if not os.path.exists(wb_csv):
        with zipfile.ZipFile(wb_csv + ".zip") as z:
            z.extract("all_waybill_info_meituan_0322.csv", data_dir)
    wb = pd.read_csv(wb_csv, index_col=0)
    dw = pd.read_csv(os.path.join(data_dir, "dispatch_waybill_meituan.csv"), index_col=0)
    dr = pd.read_csv(os.path.join(data_dir, "dispatch_rider_meituan.csv"), index_col=0)
    cw = pd.read_csv(os.path.join(data_dir, "courier_wave_info_meituan.csv"))
    dr["onhand"] = dr["courier_waybills"].apply(ast.literal_eval)   # these are ORDER ids (checked: 99.4% match)
    return wb, dw, dr, cw


# ----------------------------------------------------------------------------- geometry

LAT0, LNG0 = 45.88, 174.55          # rough centre of the (shifted) city, degrees


def to_xy(lat_e6, lng_e6):
    """Shifted lat/lng (degrees x 1e6) -> local planar km. Errors vs haversine < 0.5% at city scale."""
    lat = np.asarray(lat_e6, dtype=float) / 1e6
    lng = np.asarray(lng_e6, dtype=float) / 1e6
    x = (lng - LNG0) * 111.320 * np.cos(np.radians(LAT0))
    y = (lat - LAT0) * 110.574
    return x, y


def order_table(wb):
    """One row per order: locations (km), promised time, meal-ready time, and the accepted waybill's timestamps."""
    first = wb.sort_values("waybill_id").drop_duplicates("order_id", keep="first").set_index("order_id")
    o = pd.DataFrame(index=first.index)
    o["px"], o["py"] = to_xy(first.sender_lat, first.sender_lng)
    o["dx"], o["dy"] = to_xy(first.recipient_lat, first.recipient_lng)
    o["eta"] = first.estimate_arrived_time
    o["emr"] = first.estimate_meal_prepare_time
    o["placed"] = first.platform_order_time
    o["poi"] = first.poi_id
    o["prebook"] = first.is_prebook
    acc = wb[wb.is_courier_grabbed == 1].sort_values("waybill_id").drop_duplicates("order_id", keep="last").set_index("order_id")
    o["acc_courier"] = acc.courier_id.reindex(o.index)
    o["fetch"] = acc.fetch_time.reindex(o.index)
    o["arrive"] = acc.arrive_time.reindex(o.index)
    o["grab"] = acc.grab_time.reindex(o.index)
    return o


# ----------------------------------------------------------------------------- calibration


def calibrate_travel(wb, cw, orders, hours=None):
    """Fit  delivery minutes = handover + distance / speed  on single-order waves (a direct pickup -> drop trip).
    hours=(start, end) keeps trips picked up in that local-time window (UTC+8), e.g. (11, 13) for the noon peak."""
    single = cw[cw.order_ids.str.count(",") == 0].copy()
    single["order_id"] = single.order_ids.str.strip("[]").astype(int)
    s = orders.reindex(single.order_id).dropna(subset=["fetch", "arrive"])
    s = s[(s.fetch > 0) & (s.arrive > s.fetch)]
    if hours is not None:
        h = ((s.fetch + 8 * 3600) % 86400) / 3600
        s = s[(h >= hours[0]) & (h < hours[1])]
    d = np.hypot(s.px - s.dx, s.py - s.dy)
    t = (s.arrive - s.fetch) / 60
    keep = (d > 0.3) & (d < 6) & (t > 1) & (t < 60)
    d, t = d[keep].values, t[keep].values
    # robust line fit: least squares on the central 90% of residuals, two passes
    A = np.c_[np.ones_like(d), d]
    coef = np.linalg.lstsq(A, t, rcond=None)[0]
    for _ in range(2):
        r = t - A @ coef
        lo, hi = np.quantile(r, [0.05, 0.95])
        m = (r > lo) & (r < hi)
        coef = np.linalg.lstsq(A[m], t[m], rcond=None)[0]
    handover, min_per_km = coef
    return dict(handover_min=float(handover), speed_kmh=float(60 / min_per_km), n=int(len(d)))


# ----------------------------------------------------------------------------- epoch instances


class Epoch:
    """One dispatch checkpoint: waiting orders, candidate couriers with their current routes, Meituan's actual choice."""

    def __init__(self, t0, orders_df, couriers_df, meituan, travel):
        self.t0 = int(t0)
        self.orders = orders_df            # index order_id, columns px py dx dy eta emr poi
        self.couriers = couriers_df        # index courier_id, columns x y onhand(list of order ids)
        self.meituan = meituan             # dict courier_id -> [order ids in dispatch order]
        self.travel = travel
        self.routes = {}                   # courier_id -> Route


def build_epoch(t0, wb, dw, dr, orders, travel, window=180):
    """Orders Meituan dispatched within `window` s of checkpoint t0 to a listed candidate courier, plus all candidates."""
    w = dw[dw.dispatch_time == t0]
    cand = dr[dr.dispatch_time == t0].drop_duplicates("courier_id").set_index("courier_id")
    sub = wb[wb.order_id.isin(w.order_id)]
    sub = sub[(sub.dispatch_time >= t0) & (sub.dispatch_time <= t0 + window)].sort_values(["dispatch_time", "waybill_id"])
    first = sub.drop_duplicates("order_id", keep="first")
    first = first[first.courier_id.isin(cand.index)]
    meituan = {}
    for c, g in first.groupby("courier_id", sort=False):
        meituan[int(c)] = [int(x) for x in g.order_id]
    od = orders.loc[first.order_id, ["px", "py", "dx", "dy", "eta", "emr", "poi", "placed"]].copy()
    od["meituan_courier"] = first.set_index("order_id").courier_id.reindex(od.index).astype(int)
    od["meituan_grabbed"] = first.set_index("order_id").is_courier_grabbed.reindex(od.index).astype(int)
    cx, cy = to_xy(cand.rider_lat, cand.rider_lng)
    cd = pd.DataFrame({"x": cx, "y": cy, "onhand": cand.onhand}, index=cand.index)
    ep = Epoch(t0, od, cd, meituan, travel)
    ep.n_waiting = len(w)
    return ep


# ----------------------------------------------------------------------------- routes

class Route:
    """A courier's remaining stops. kind: +1 pickup, -1 drop. Times in minutes after the checkpoint."""
    __slots__ = ("x", "y", "kind", "oid", "ready", "due", "dist", "late", "nlate")

    def __init__(self, x0, y0):
        self.x = [x0]; self.y = [y0]; self.kind = [0]; self.oid = [-1]
        self.ready = [0.0]; self.due = [np.inf]
        self.dist = 0.0; self.late = 0.0; self.nlate = 0

    def copy(self):
        r = Route.__new__(Route)
        r.x = self.x[:]; r.y = self.y[:]; r.kind = self.kind[:]; r.oid = self.oid[:]
        r.ready = self.ready[:]; r.due = self.due[:]
        r.dist = self.dist; r.late = self.late; r.nlate = self.nlate
        return r


def evaluate(r, speed_kmh, service_min):
    """Forward pass: total km, total late minutes and number of late drops (no buffer)."""
    sp, sd = service_min if isinstance(service_min, tuple) else (service_min, service_min)
    t, dist, late, nl = 0.0, 0.0, 0.0, 0
    x, y = r.x[0], r.y[0]
    k = 60.0 / speed_kmh
    for i in range(1, len(r.x)):
        d = ((r.x[i] - x) ** 2 + (r.y[i] - y) ** 2) ** 0.5
        dist += d
        t += d * k
        if r.kind[i] == 1:
            if t < r.ready[i]:
                t = r.ready[i]
            t += sp
        else:
            t += sd
            if t > r.due[i]:
                late += t - r.due[i]; nl += 1
        x, y = r.x[i], r.y[i]
    return dist, late, nl


def insert_order(r, ox, oy, qx, qy, ready, due, oid, speed_kmh, service_min, lam, topk=4, pickup=True):
    """Cheapest insertion of pickup (ox,oy) and drop (qx,qy) into route r, precedence kept.
    Candidate positions are ranked by added km; the best `topk` are re-scored on km + lam * late minutes.
    Returns the new route (evaluated)."""
    X = np.asarray(r.x); Y = np.asarray(r.y)
    n = len(X)                                  # positions: insert after index i (0..n-1)
    nxtX = np.r_[X[1:], np.nan]; nxtY = np.r_[Y[1:], np.nan]
    seg = np.hypot(nxtX - X, nxtY - Y)
    seg = np.where(np.isnan(seg), 0.0, seg)
    end = np.isnan(nxtX)
    if not pickup:                               # drop only (order already in hand)
        dd = np.hypot(X - qx, Y - qy) + np.where(end, 0.0, np.hypot(nxtX - qx, nxtY - qy)) - seg
        cand = [(-1, int(j)) for j in np.argsort(dd)[:topk]]
    else:
        ins_p = np.hypot(X - ox, Y - oy) + np.where(end, 0.0, np.hypot(nxtX - ox, nxtY - oy)) - seg
        ins_d = np.hypot(X - qx, Y - qy) + np.where(end, 0.0, np.hypot(nxtX - qx, nxtY - qy)) - seg
        same = np.hypot(X - ox, Y - oy) + np.hypot(ox - qx, oy - qy) + np.where(end, 0.0, np.hypot(nxtX - qx, nxtY - qy)) - seg
        M = ins_p[:, None] + ins_d[None, :]
        M[np.tril_indices(n, -1)] = np.inf
        M[np.diag_indices(n)] = same
        flat = np.argsort(M, axis=None)[:topk]
        cand = [(int(f // n), int(f % n)) for f in flat]
    best, bestv = None, np.inf
    for i, j in cand:
        nr = r.copy()
        if pickup:
            # insert drop first (after j), then pickup (after i); j >= i keeps precedence
            for (pos, px, py, kd, rd, du) in ((j, qx, qy, -1, 0.0, due), (i, ox, oy, 1, ready, np.inf)):
                nr.x.insert(pos + 1, px); nr.y.insert(pos + 1, py); nr.kind.insert(pos + 1, kd)
                nr.oid.insert(pos + 1, oid); nr.ready.insert(pos + 1, rd); nr.due.insert(pos + 1, du)
        else:
            nr.x.insert(j + 1, qx); nr.y.insert(j + 1, qy); nr.kind.insert(j + 1, -1)
            nr.oid.insert(j + 1, oid); nr.ready.insert(j + 1, 0.0); nr.due.insert(j + 1, due)
        nr.dist, nr.late, nr.nlate = evaluate(nr, speed_kmh, service_min)
        v = nr.dist + lam * nr.late
        if v < bestv:
            best, bestv = nr, v
    return best


def courier_route(ep, c, orders_all, lam=1.0):
    """Rebuild courier c's current route from its on-hand orders (picked-up orders need only a drop)."""
    if c in ep.routes:
        return ep.routes[c]
    tr = ep.travel
    row = ep.couriers.loc[c]
    r = Route(row.x, row.y)
    oh = [o for o in row.onhand if o in orders_all.index]
    oh.sort(key=lambda o: orders_all.at[o, "eta"])
    for o in oh:
        a = orders_all.loc[o]
        picked = (a.fetch > 0) and (a.fetch <= ep.t0)
        r = insert_order(r, a.px, a.py, a.dx, a.dy, (a.emr - ep.t0) / 60, (a.eta - ep.t0) / 60, int(o),
                         tr["speed_kmh"], tr["service_min"], tr.get("lam_ins", lam), topk=tr.get("topk", 4), pickup=not picked)
    if len(r.x) == 1:
        r.dist, r.late, r.nlate = 0.0, 0.0, 0
    ep.routes[c] = r
    return r


def drop_times(r, speed_kmh, service_min):
    """Predicted delivery time (minutes after checkpoint) of every drop on route r."""
    sp, sd = service_min if isinstance(service_min, tuple) else (service_min, service_min)
    t, x, y, out = 0.0, r.x[0], r.y[0], {}
    k = 60.0 / speed_kmh
    for i in range(1, len(r.x)):
        d = ((r.x[i] - x) ** 2 + (r.y[i] - y) ** 2) ** 0.5
        t += d * k
        if r.kind[i] == 1:
            t = max(t, r.ready[i]) + sp
        else:
            t += sd
            out[r.oid[i]] = t
        x, y = r.x[i], r.y[i]
    return out


def validate(epochs, orders_all, travel, include_onhand=False):
    """Simulate Meituan's actual dispatches and compare predicted with actual delivery times (accepted orders only)."""
    rows = []
    for ep in epochs:
        ep.travel = travel
        ep.routes = {}
        for c, b in ep.meituan.items():
            r = courier_route(ep, c, orders_all)
            for o in b:
                a = ep.orders.loc[o]
                r = insert_order(r, a.px, a.py, a.dx, a.dy, (a.emr - ep.t0) / 60, (a.eta - ep.t0) / 60, int(o),
                                 travel["speed_kmh"], travel["service_min"], travel.get("lam_ins", 1.0), topk=travel.get("topk", 4))
            dt = drop_times(r, travel["speed_kmh"], travel["service_min"])
            for o in dt:
                if orders_all.at[o, "acc_courier"] == c and orders_all.at[o, "arrive"] > 0:
                    rows.append((ep.t0, o, o in b, dt[o], (orders_all.at[o, "arrive"] - ep.t0) / 60, (orders_all.at[o, "eta"] - ep.t0) / 60))
        ep.routes = {}
    v = pd.DataFrame(rows, columns=["t0", "order_id", "new", "pred", "actual", "due"])
    return v if include_onhand else v[v.new]


def with_dwell(travel, dwell_min):
    """Travel settings with `dwell_min` extra minutes at every restaurant pickup (queue, parking, meal not quite ready)."""
    h = travel["handover_min"] / 2
    return dict(travel, dwell_min=dwell_min, service_min=(h + dwell_min, h))


def fit_pickup_dwell(epochs, orders_all, travel, lo=0.0, hi=15.0, iters=8):
    """Bisection on the restaurant dwell so that predicted = actual mean delivery time of newly dispatched orders."""
    for _ in range(iters):
        mid = (lo + hi) / 2
        v = validate(epochs, orders_all, with_dwell(travel, mid))
        if (v.pred - v.actual).mean() < 0:      # predictions too fast -> more dwell
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def fit_effective_speed(epochs, orders_all, travel, lo=5.0, hi=None, iters=7):
    """Bisection on speed so that mean predicted delivery time equals the actual mean on `epochs`."""
    hi = hi or travel["speed_kmh"]
    for _ in range(iters):
        mid = (lo + hi) / 2
        v = validate(epochs, orders_all, dict(travel, speed_kmh=mid))
        if (v.pred - v.actual).mean() > 0:      # too slow -> raise speed
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def bundle_cost(ep, c, bundle, orders_all, lam):
    """Added km, added late minutes and new late drops when courier c takes `bundle` (tuple of order ids).
    Non-additive: tries every insertion sequence for bundles up to 3 orders."""
    base = courier_route(ep, c, orders_all)
    tr = ep.travel
    best = None
    seqs = itertools.permutations(bundle) if len(bundle) <= 3 else [tuple(sorted(bundle, key=lambda o: ep.orders.at[o, "eta"]))]
    for seq in seqs:
        r = base
        for o in seq:
            a = ep.orders.loc[o]
            r = insert_order(r, a.px, a.py, a.dx, a.dy, (a.emr - ep.t0) / 60, (a.eta - ep.t0) / 60, int(o),
                             tr["speed_kmh"], tr["service_min"], tr.get("lam_ins", lam), topk=tr.get("topk", 4))
        v = (r.dist - base.dist) + lam * (r.late - base.late)
        if best is None or v < best[0]:
            best = (v, r.dist - base.dist, r.late - base.late, r.nlate - base.nlate)
    return best[1], best[2], best[3]
