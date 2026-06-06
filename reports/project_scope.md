# Project Scope (Phase 1)

## Objective

A dynamic portfolio allocator that controls downside tail risk via an explicit
CVaR constraint, allocating across a liquid multi-asset universe net of costs,
benchmarked against deterministic optimisers and unconstrained RL.

## First decisions (v1)

| Decision | Choice for v1 | Rationale |
| --- | --- | --- |
| Universe | 5 macro asset classes: equity, rates, credit, commodity, FX (`configs/universe.yaml`) | Liquid, diversifying, small enough to train quickly. ETF/futures proxies slot into the same schema. |
| Data | Synthetic regime-switching panel | Lets the whole pipeline run offline and end-to-end; real loaders plug into the same contract. |
| Rebalance frequency | Weekly | Matches the TODO v1 target; daily is a later extension. |
| Portfolio type | Liability-free, long-only | Simplest meaningful setting; benchmark-relative is a later variant. |
| Constraints | Long-only, max weight, turnover cap, gross cap (cash floor optional) | Enforced by projection in `crlpa/envs/constraints.py`. |
| Primary risk constraint | Rolling historical CVaR limit (95%, weekly) | Model-estimated CVaR is a later option; the safety critic already learns a CVaR cost-to-go. |
| Success claim | (1) Reduce CVaR / breaches vs. unconstrained RL; (2) stay competitive with optimisers after costs | Evaluated with paired block-bootstrap CIs on the test split. |

## Splits

Chronological train/validation/test (default 60/20/20) with no shuffling.
Observations at decision step `t` use only returns strictly before `t`
(enforced by `tests/test_no_lookahead.py`).

## Baselines

Deterministic: cash, equal weight, inverse vol, minimum variance, mean-variance,
risk parity, minimum-CVaR (Rockafellar–Uryasev LP). RL: unconstrained
actor-critic (same code path, `constrained=False`).

## Deliverable status

The acceptance-criteria items that are runnable today — leak-free data, a
reproducible weekly clock, tested environment and constraints, transaction costs,
unconstrained and CVaR-constrained RL, metrics with CVaR/drawdown/breach
reporting, and bootstrap significance — are implemented. Walk-forward, robustness,
and ablation suites are scoped against the same interfaces (see `TODO.md`).
