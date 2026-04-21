# Search latency runbook

## Sudden spike in hybrid search latency

1. **Confirm scope**: verify whether slowness affects `lens focus` only, semantic path only, or both; note approximate p95 versus the prior week from your metrics or a few timed CLI runs.
2. **Isolate the bottleneck**: check disk I/O and CPU on the host running the index; if lexical search is fast but semantic is slow, suspect model load or MLX contention from other processes.
3. **Reduce load**: stop ad-hoc bulk queries, pause any automated scripts hitting `lens focus`, and ask users to avoid wide “render” batches until latency recovers.
4. **Communicate**: post in `#incidents` with impact (interactive search vs. batch), affected workflows, and whether a short maintenance window is needed.
5. **Post-incident**: document the bottleneck class (I/O, CPU, model, config); file tickets for caching, index layout, or capacity if the root cause is recurring.

## Index corruption or missing chunks after a deploy

1. **Confirm**: run a small known query that should hit specific files; compare results to a backup checkout or prior export if available.
2. **Stop writes**: avoid `lens calibrate` and corpus edits until you know whether the index or the source tree is wrong.
3. **Rebuild**: from a clean state, run `lens calibrate` after verifying `config.yaml` paths; validate spot-check queries before announcing all-clear.
4. **Communicate**: state whether users must re-run calibrate locally and whether any cached vectors were invalidated.

## On-call expectations

Primary owns mitigation and comms; secondary pulls logs and reproduces on a second machine. Escalate to the platform owner if p95 interactive search stays above agreed SLO for more than thirty minutes after standard mitigations.
