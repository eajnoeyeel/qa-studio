---
doc_id: help_sso_config
title: SSO Configuration Guide
version: v1
tags: [login_sso, workspace_access]
source_url: https://help.example.com/sso/configuration
---

# SSO Configuration Guide

## Supported Identity Providers

- Okta
- Azure Active Directory
- Google Workspace
- OneLogin
- Custom SAML 2.0

## Prerequisites

1. Enterprise or Team plan
2. Workspace owner or admin role
3. Domain verification completed
4. IdP admin credentials

## Setup Steps

### Step 1: Verify Domain
1. Go to Settings → Security → Domain Management
2. Add your domain (e.g., company.com)
3. Add TXT record to DNS
4. Click "Verify Domain"
5. Wait up to 48 hours for propagation

### Step 2: Configure IdP
1. In your IdP, create new SAML application
2. Use our metadata URL or manual config:
   - Entity ID: `https://app.example.com/saml/{workspace_id}`
   - ACS URL: `https://app.example.com/saml/callback`
   - NameID: Email address

### Step 3: Configure in App
1. Go to Settings → Security → SSO
2. Enter IdP metadata URL or upload XML
3. Map attributes (email required)
4. Test connection
5. Enable SSO

## Common Error Codes

| Code | Meaning | Solution |
|------|---------|----------|
| SAML_001 | Invalid assertion | Check IdP time sync |
| SAML_002 | User not provisioned | Add user in IdP |
| SAML_003 | Domain not verified | Complete domain verification |
| SAML_004 | Certificate expired | Update IdP certificate |

## Troubleshooting

**Users can't log in:**
1. Verify user exists in IdP
2. Check attribute mapping
3. Ensure email matches

**"Domain not found" error:**
1. Confirm domain verification
2. Check DNS propagation
3. Wait 24 hours after changes

**Admin lockout:**
- Owners can always use email/password
- Use backup admin account
- Contact support for recovery
