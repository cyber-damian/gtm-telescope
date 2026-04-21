# Render and grounding runbook

## Answers drift from source or cite wrong snippets

1. **Confirm scope**: reproduce with `lens render` on a fixed query; capture whether `lens focus` for the same query returns the expected passages.
2. **Check grounding path**: if focus is correct but render is not, inspect prompt and model settings in config; ensure the render step is actually receiving the focused context bundle.
3. **Degrade safely**: switch high-stakes workflows to “focus only” (human reads snippets) until grounding is verified; disable any auto-posting of rendered answers externally.
4. **Communicate**: post in `#incidents` with examples (redacted) of bad vs. expected behavior; state whether customer-facing channels are affected.
5. **Post-incident**: capture whether the failure was retrieval, context truncation, or model behavior; add regression tests or golden queries where feasible.

## API keys or model provider errors

1. **Classify the error**: distinguish rate limits, invalid credentials, and regional outages from provider status pages or error payloads.
2. **Rotate if needed**: if a key may be exposed, revoke and rotate in the secret store; update `.env` only through your team’s approved process—never commit secrets.
3. **Fallback**: where policy allows, fall back to focus-only mode or a backup model profile defined in config; document which customers or workflows use the fallback.
4. **Communicate**: give a clear ETA based on provider recovery or key propagation delay.

## On-call expectations

Primary handles reproduction and temporary degradation; secondary verifies secrets and config in staging. Escalate if any suspected credential leak cannot be ruled out within fifteen minutes.
