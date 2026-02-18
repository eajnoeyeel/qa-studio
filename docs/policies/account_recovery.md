---
doc_id: policy_account_recovery
title: Account Recovery Policy
version: v1
tags: [login_sso, workspace_access, security]
---

# Account Recovery Policy

## Standard Account Recovery

### Lost Password
1. Direct to "Forgot Password" link on login page
2. Recovery email sent to registered address
3. Link expires in 24 hours
4. Maximum 5 attempts per hour

### Lost 2FA Device
1. Verify identity via recovery email
2. Provide backup codes if saved
3. If no backup: escalate to security team
4. 48-hour cooling period for 2FA reset

## SSO Account Issues

### Cannot Access via SSO
1. Verify IdP (Identity Provider) status with IT admin
2. Check if user is provisioned in IdP
3. Verify domain claim is correct
4. Common error codes:
   - `SAML_001`: Invalid assertion
   - `SAML_002`: User not provisioned
   - `SAML_003`: Domain not verified

### SSO Admin Requirements
For SSO configuration changes:
- Only workspace owners or IT admins
- Requires verification of admin status
- Changes take up to 1 hour to propagate

## Locked Accounts

Accounts are locked after:
- 10 failed login attempts
- Suspicious activity detection
- Billing payment failure (downgrade, not lock)

### Unlock Process
1. Wait 30 minutes for automatic unlock
2. Or use "Unlock Account" email
3. Contact support if email not received

## Deleted Accounts

### Recently Deleted (< 30 days)
- Can be restored by workspace admin
- All data intact
- Settings preserved

### Permanently Deleted (> 30 days)
- Data cannot be recovered
- Username becomes available
- Requires new account creation
