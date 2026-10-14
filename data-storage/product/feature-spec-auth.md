# Feature spec: passwordless sign-in

## Goal

Let returning users sign in with a magic link emailed to their verified address, reducing password fatigue and support tickets for resets.

## Security

Links expire after fifteen minutes and are single-use. Rate-limit requests per email and per IP. Log authentication outcomes for fraud review without storing raw tokens.

## Rollout

Ship behind a feature flag for internal dogfood first, then 5% of new signups before general availability.

## Enterprise SSO and provisioning

SAML 2.0 and OIDC single sign-on are supported today for Okta, Microsoft Entra ID and Google Workspace. SCIM 2.0 user provisioning is not available yet and is on the roadmap for the second half of the year; until then, accounts are created on first SSO login and deactivated by an admin in the console.
