# Method and implementation map

[← Project homepage](../README.md)

The [paper](../assets/paper.pdf) is the reference for the bilevel formulation, assumptions, and solver. This page connects the implementation to the paper; it does not replace its mathematical definitions.

## Reading order

1. [`ray_trainer.py`](../OPDVR/verl/verl/trainer/ppo/ray_trainer.py): student rollouts, teacher/outcome signals, and the weighted training loop.
2. [`dp_actor.py`](../OPDVR/verl/verl/workers/actor/dp_actor.py): per-token credit construction; search for `compute_credit`.
3. [`jvp_influence.py`](../OPDVR/verl/verl/workers/actor/jvp_influence.py): directional derivatives computed with a JVP.
4. [`credit_gate.py`](../OPDVR/verl/verl/workers/actor/credit_gate.py): shadow optimizer direction and the `iw_gate` weight transformation.

## Default gate

For a sampled response token, the teacher push is the teacher–student log-probability difference. Its product with the directional derivative along the stored reward-improving direction gives the implemented credit. On the active set, the gate uses:

$$
w_t = \operatorname{clip}(1 + \lambda z_t,\; 0.001,\; 3),
\qquad z_t = c_t / \operatorname{RMS}_{\mathcal S}(c).
$$

Weights outside the active set remain 1. The root launcher's active fraction is 1.0; groups without a reward difference provide no group-relative direction. Before a usable direction exists, the implementation falls back to unit weights. See the actual code for masks and numerical handling.

The default preset uses `credit_mode=jvp` with `credit_gate_kind=iw`. The implementation also supports finite-difference credit computation and a binary sign gate.

Token credits are context- and policy-dependent local signals, not intrinsic scores assigned permanently to token types.
