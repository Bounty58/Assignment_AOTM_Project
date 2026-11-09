"""Column generation (courier 'bids' on order bundles), the Pyomo set-partitioning model, and plan evaluation."""
import itertools, time
import numpy as np
import pandas as pd
import pyomo.environ as pyo
from dispatch_core import courier_route, insert_order, drop_times


# ----------------------------------------------------------------------------- columns

def _order_attrs(ep):
    o = ep.orders
    t0 = ep.t0
    return {int(i): (r.px, r.py, r.dx, r.dy, (r.emr - t0) / 60, (r.eta - t0) / 60)
            for i, r in zip(o.index, o.itertuples())}


def _cost(ep, c, bundle, attrs, lam, orders_all):
    """Best insertion sequence for `bundle` on courier c.
    Returns (added km, added late min [all orders], added late drops, late min of the NEW orders, new orders late)."""
    base = courier_route(ep, c, orders_all)
    tr = ep.travel
    best = None
    for seq in itertools.permutations(bundle):
        r = base
        for o in seq:
            px, py, dx, dy, rd, du = attrs[o]
            r = insert_order(r, px, py, dx, dy, rd, du, o, tr["speed_kmh"], tr["service_min"], lam, topk=tr.get("topk", 4))
        v = (r.dist - base.dist) + lam * (r.late - base.late)
        if best is None or v < best[0]:
            best = (v, r)
        if len(bundle) > 3:          # Meituan's large bundles: one ETA-ordered sequence is enough
            break
    r = best[1]
    dt = drop_times(r, tr["speed_kmh"], tr["service_min"])
    lat = [max(0.0, dt[o] - attrs[o][5]) for o in bundle]
    slack = tuple(round(attrs[o][5] - dt[o], 2) for o in bundle)       # promised minus predicted delivery time
    return r.dist - base.dist, r.late - base.late, r.nlate - base.nlate, float(sum(lat)), int(sum(x > 0 for x in lat)), slack


def generate_columns(ep, orders_all, K=12, radius_km=3.0, cap=8, top_pairs=5, top_triples=3, lam=1.0):
    """Courier bids: singles to the K nearest eligible couriers, pairs and triples from each courier's best singles,
    plus Meituan's actual (courier, bundle) choices so that Meituan's plan is always feasible."""
    attrs = _order_attrs(ep)
    oids = np.array(list(attrs))
    PX = np.array([attrs[o][0] for o in oids]); PY = np.array([attrs[o][1] for o in oids])
    cids = ep.couriers.index.values
    CX = ep.couriers.x.values; CY = ep.couriers.y.values
    load = ep.couriers.onhand.apply(len).values
    D = np.hypot(PX[:, None] - CX[None, :], PY[:, None] - CY[None, :])     # orders x couriers, km
    D[:, load + 1 > cap] = np.inf
    cols = {}                                                              # (courier, bundle) -> (km, late, nlate)
    for i, o in enumerate(oids):
        near = np.argsort(D[i])[:K]
        for j in near:
            if D[i, j] <= radius_km:
                c = int(cids[j])
                cols[(c, (int(o),))] = _cost(ep, c, (int(o),), attrs, lam, orders_all)
    # bundles from each courier's cheapest singles
    by_c = {}
    for (c, b), v in cols.items():
        by_c.setdefault(c, []).append((v[0] + lam * v[1], b[0]))
    loads = dict(zip(cids, load))
    for c, lst in by_c.items():
        lst.sort()
        best = [o for _, o in lst]
        if loads[c] + 2 <= cap:
            for b in itertools.combinations(best[:top_pairs], 2):
                cols[(c, tuple(sorted(b)))] = _cost(ep, c, b, attrs, lam, orders_all)
        if loads[c] + 3 <= cap:
            for b in itertools.combinations(best[:top_triples], 3):
                cols[(c, tuple(sorted(b)))] = _cost(ep, c, b, attrs, lam, orders_all)
    # Meituan's actual choices (any size), flagged
    meituan_keys = []
    for c, b in ep.meituan.items():
        key = (int(c), tuple(sorted(b)))
        if key not in cols:
            cols[key] = _cost(ep, int(c), tuple(b), attrs, lam, orders_all)
        meituan_keys.append(key)
    cdf = pd.DataFrame([(c, b, len(b), *v) for (c, b), v in cols.items()],
                       columns=["courier", "bundle", "size", "km", "late", "nlate", "late_new", "n_new_late", "slack"])
    cdf["late_onhand"] = cdf.late - cdf.late_new          # extra lateness pushed onto orders already in the bag
    cdf["idx"] = np.arange(len(cdf))
    key_to_idx = {(c, b): i for i, (c, b) in enumerate(zip(cdf.courier, cdf.bundle))}
    cdf["meituan"] = False
    cdf.loc[[key_to_idx[k] for k in meituan_keys], "meituan"] = True
    return cdf


# ----------------------------------------------------------------------------- acceptance model

ACC_FEATURES = ["dist_cm", "onhand", "idle", "km", "late"]


def acceptance_features(ep, cdf):
    """Per (column, order) features for the courier-acceptance model."""
    cx = ep.couriers.x; cy = ep.couriers.y; oh = ep.couriers.onhand.apply(len)
    rows = []
    for idx, c, b, km, late in zip(cdf.idx, cdf.courier, cdf.bundle, cdf.km, cdf.late):
        for o in b:
            d = np.hypot(cx[c] - ep.orders.at[o, "px"], cy[c] - ep.orders.at[o, "py"])
            rows.append((idx, o, d, oh[c], int(oh[c] == 0), km / len(b), late / len(b)))
    return pd.DataFrame(rows, columns=["idx", "order_id"] + ACC_FEATURES)


def fit_logit(X, y, l2=1e-3, iters=50):
    """Plain Newton/IRLS logistic regression (keeps the notebook free of extra dependencies)."""
    X = np.c_[np.ones(len(X)), X]
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ w))
        g = X.T @ (p - y) + l2 * w
        H = (X * (p * (1 - p))[:, None]).T @ X + l2 * np.eye(len(w))
        step = np.linalg.solve(H, g)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return w


def predict_logit(w, X):
    return 1 / (1 + np.exp(-(np.c_[np.ones(len(X)), X] @ w)))


def auc(y, p):
    order = np.argsort(p)
    r = np.empty(len(p)); r[order] = np.arange(1, len(p) + 1)
    n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


# ----------------------------------------------------------------------------- Pyomo model

def solve_dispatch(cdf, orders, couriers, lam=1.0, mu=0.0, exp_reject=None, max_size=None, unassigned_penalty=None,
                   time_limit=60):
    """Set partitioning / combinatorial-auction winner determination in Pyomo, solved with HiGHS.

    min  sum_j (km_j + lam*late_j + mu*E[rejections]_j) z_j  (+ penalty * u_o)
    s.t. sum_{j contains o} z_j (+ u_o) = 1      for every order o
         sum_{j of courier c} z_j <= 1           for every courier c
         z_j binary
    """
    use = cdf if max_size is None else cdf[(cdf["size"] <= max_size)]
    cost = (use.km + lam * use.late).values
    if mu and exp_reject is not None:
        cost = cost + mu * exp_reject.reindex(use.idx).fillna(0).values
    J = list(use.idx)
    cost_of = dict(zip(J, cost))
    by_order, by_courier = {}, {}
    for j, c, b in zip(use.idx, use.courier, use.bundle):
        by_courier.setdefault(c, []).append(j)
        for o in b:
            by_order.setdefault(o, []).append(j)

    m = pyo.ConcreteModel("meituan_dispatch")
    m.J = pyo.Set(initialize=J)
    m.O = pyo.Set(initialize=list(orders))
    m.C = pyo.Set(initialize=list(by_courier))
    m.z = pyo.Var(m.J, domain=pyo.Binary)
    pen = unassigned_penalty
    if pen is not None:
        m.u = pyo.Var(m.O, domain=pyo.Binary)
    m.obj = pyo.Objective(expr=sum(cost_of[j] * m.z[j] for j in m.J) + (sum(pen * m.u[o] for o in m.O) if pen is not None else 0),
                          sense=pyo.minimize)
    m.cover = pyo.Constraint(m.O, rule=lambda m, o: sum(m.z[j] for j in by_order.get(o, [])) + (m.u[o] if pen is not None else 0) == 1)
    m.one_bundle = pyo.Constraint(m.C, rule=lambda m, c: sum(m.z[j] for j in by_courier[c]) <= 1)

    t = time.time()
    solver = pyo.SolverFactory("appsi_highs")
    solver.config.time_limit = time_limit
    res = solver.solve(m)
    secs = time.time() - t
    chosen = [j for j in J if pyo.value(m.z[j]) > 0.5]
    unassigned = [o for o in m.O if pen is not None and pyo.value(m.u[o]) > 0.5]
    return dict(chosen=chosen, unassigned=unassigned, seconds=secs, objective=pyo.value(m.obj),
                n_vars=len(J), n_cons=len(m.O) + len(m.C), status=str(res.solver.termination_condition))


class DispatchModel:
    """The same set-partitioning model, built once per epoch with a mutable cost per column,
    so the weight sweeps re-solve in place instead of rebuilding the model."""

    def __init__(self, cdf, orders, max_size=None, unassigned_penalty=None, time_limit=60):
        use = cdf if max_size is None else cdf[cdf["size"] <= max_size]
        self.cdf = use
        J = list(use.idx)
        by_order, by_courier = {}, {}
        for j, c, b in zip(use.idx, use.courier, use.bundle):
            by_courier.setdefault(c, []).append(j)
            for o in b:
                by_order.setdefault(o, []).append(j)
        m = pyo.ConcreteModel("meituan_dispatch")
        m.J = pyo.Set(initialize=J)
        m.O = pyo.Set(initialize=list(orders))
        m.C = pyo.Set(initialize=list(by_courier))
        m.cost = pyo.Param(m.J, initialize=dict(zip(J, use.km.values)), mutable=True)
        m.z = pyo.Var(m.J, domain=pyo.Binary)
        pen = unassigned_penalty
        if pen is not None:
            m.u = pyo.Var(m.O, domain=pyo.Binary)
        m.obj = pyo.Objective(expr=sum(m.cost[j] * m.z[j] for j in m.J) + (sum(pen * m.u[o] for o in m.O) if pen is not None else 0),
                              sense=pyo.minimize)
        m.cover = pyo.Constraint(m.O, rule=lambda m, o: sum(m.z[j] for j in by_order.get(o, [])) + (m.u[o] if pen is not None else 0) == 1)
        m.one_bundle = pyo.Constraint(m.C, rule=lambda m, c: sum(m.z[j] for j in by_courier[c]) <= 1)
        self.m, self.J, self.pen = m, J, pen
        self.solver = pyo.SolverFactory("appsi_highs")
        self.solver.config.time_limit = time_limit
        self.n_vars, self.n_cons = len(J) + (len(m.O) if pen is not None else 0), len(m.O) + len(m.C)

    def solve(self, lam=1.0, mu=0.0, exp_reject=None):
        cost = self.cdf.km.values + lam * self.cdf.late.values
        if mu and exp_reject is not None:
            cost = cost + mu * exp_reject.reindex(self.cdf.idx).fillna(0).values
        for j, c in zip(self.J, cost):
            self.m.cost[j] = float(c)
        t = time.time()
        res = self.solver.solve(self.m)
        secs = time.time() - t
        chosen = [j for j in self.J if pyo.value(self.m.z[j]) > 0.5]
        unassigned = [o for o in self.m.O if self.pen is not None and pyo.value(self.m.u[o]) > 0.5]
        return dict(chosen=chosen, unassigned=unassigned, seconds=secs, objective=pyo.value(self.m.obj),
                    status=str(res.solver.termination_condition))


def fit_late_model(v):
    """Calibrated P(new order delivered late | predicted slack), fitted on validation rows (newly dispatched orders).
    Features: slack and its negative part, so the curve can bend at zero."""
    s = (v.due - v.pred).values
    X = np.c_[s, np.minimum(s, 0)]
    return fit_logit(X, (v.actual > v.due).values.astype(float))


def late_prob(w, slack):
    s = np.asarray(slack, dtype=float)
    return predict_logit(w, np.c_[s, np.minimum(s, 0)])


def plan_summary(cdf, chosen, p_accept=None, late_w=None):
    """Totals for a plan given chosen column indices."""
    sel = cdf.set_index("idx").loc[chosen]
    out = dict(orders=int(sel["size"].sum()), couriers=len(sel), km=float(sel.km.sum()), late=float(sel.late.sum()),
               late_new=float(sel.late_new.sum()), late_onhand=float(sel.late_onhand.sum()),
               new_orders_late=int(sel.n_new_late.sum()), new_late_drops=int(sel.nlate.sum()),
               multi_share=float(sel.loc[sel["size"] > 1, "size"].sum() / sel["size"].sum()))
    if p_accept is not None:
        pa = p_accept[p_accept.idx.isin(chosen)]
        out["exp_accept"] = float(pa.p.mean())
    if late_w is not None:
        sl = [x for t in sel.slack for x in t]
        out["exp_late_share"] = float(late_prob(late_w, sl).mean())
    return out
