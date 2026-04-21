# Document ingestion runbook

## `lens calibrate` fails or hangs on a subset of files

1. **Confirm scope**: note file types and paths that fail; check whether failures are new after a corpus drop or a tool upgrade.
2. **Isolate bad inputs**: move suspect files to a quarantine folder outside the indexed tree; re-run calibrate on the remainder to restore a usable index.
3. **Check converters**: for PDF or Office paths, confirm optional converter dependencies and disk space; capture stack traces from a single-file retry in a dev shell.
4. **Communicate**: tell the team which directories are excluded and when a full re-index is expected to complete.
5. **Post-incident**: add file-type or size limits to the intake checklist if the failure class is repeatable; link fixed upstream issues in your tracker.

## Sudden growth in storage or cache under `data-storage`

1. **Measure**: identify largest directories (extracted markdown, caches, attachments) and whether growth aligns with a bulk import.
2. **Trim safely**: remove only caches and regenerable artifacts per your retention policy; do not delete the only copy of source documents without backup confirmation.
3. **Throttle intake**: pause large batch imports until quotas or disk are addressed; coordinate with owners of shared volumes.
4. **Communicate**: state current disk headroom and any temporary pause on new uploads.

## On-call expectations

Primary drives triage and partial index recovery; secondary validates backups before any destructive cleanup. Escalate to storage or IT if the host volume is above critical threshold and cannot be freed quickly from regenerable data.
