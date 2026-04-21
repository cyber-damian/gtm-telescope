# Incident response runbook

## Database outage during peak traffic

1. **Confirm scope**: check dashboards for which regions or shards show elevated error rates and latency.
2. **Failover**: if the primary writer is unhealthy, promote the warm standby and update the connection string in the edge config service. Expect a few minutes of read skew.
3. **Throttle non-critical work**: pause batch reindex jobs and lower background sync concurrency so connections return to the pool for user-facing traffic.
4. **Communicate**: post in `#incidents` with current customer impact, ETA, and who owns the decision to degrade features (e.g. disable recommendations block).
5. **Post-incident**: schedule a blameless review within five business days; capture timeline, detection gaps, and follow-up tickets.

## On-call expectations

Primary carries the pager; secondary shadows for handoffs. Escalate to infra lead if failover does not restore p95 read latency under 200ms within fifteen minutes.
