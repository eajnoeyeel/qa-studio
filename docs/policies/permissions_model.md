---
doc_id: policy_permissions
title: Permission and Sharing Model
version: v1
tags: [permission_sharing, workspace_access, all]
---

# Permission and Sharing Model

## Workspace Roles

| Role | Capabilities |
|------|-------------|
| Owner | Full control, billing, delete workspace |
| Admin | Manage members, settings, integrations |
| Member | Create content, invite guests |
| Guest | Access only shared content |

## Page Permissions

### Permission Levels
1. **Full Access**: Edit, comment, share
2. **Can Edit**: Modify content, cannot share
3. **Can Comment**: View and comment only
4. **Can View**: Read-only access

### Inheritance
- Child pages inherit parent permissions by default
- Can override at any level
- "Lock" prevents inheritance changes

## Sharing Options

### Internal Sharing
- Share with specific members
- Share with workspace
- Share with groups

### External Sharing (if enabled)
- Public link (anyone with link)
- Email-specific access
- Expiring links (Pro+)
- Password protection (Team+)

## Common Issues

### "Access Denied" Errors
1. Check if user is workspace member
2. Verify page-level permissions
3. Check parent page inheritance
4. Confirm sharing link validity

### Cannot Change Permissions
- Only page owner or admin can change
- Check if page is locked
- Workspace setting may restrict external sharing

### Guest Access Problems
- Guest must accept invitation first
- Email must match invitation
- Guest limit may be reached (plan-dependent)

## Permission Model Mismatches

Common misunderstandings:
1. Workspace role vs page permission (different)
2. Group membership vs direct access
3. Public workspace vs public page
4. Share vs transfer ownership
