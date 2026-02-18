---
doc_id: help_permissions_guide
title: Understanding Permissions and Sharing
version: v1
tags: [permission_sharing, workspace_access]
source_url: https://help.example.com/permissions/overview
---

# Understanding Permissions and Sharing

## Permission Levels

### Workspace Level
- **Owner**: Full control, billing, can delete workspace
- **Admin**: Manage members, settings, integrations
- **Member**: Create pages, access shared content
- **Guest**: Limited access to specific pages only

### Page Level
- **Full Access**: Edit, share, delete
- **Can Edit**: Edit content, cannot share or delete
- **Can Comment**: View and comment only
- **Can View**: Read-only access

## Sharing Pages

### With Workspace Members
1. Click Share button
2. Search for team member
3. Select permission level
4. Click Invite

### With External Users (Guests)
1. Click Share
2. Enter email address
3. Select permission (limited options)
4. Guest receives invite link

### Public Sharing
1. Click Share
2. Enable "Share to web"
3. Choose: Can view OR Can comment
4. Copy link

## Permission Inheritance

- Pages inherit parent page permissions by default
- Override inheritance per page
- Inheritance can be locked by admins

## Common Issues

### "Cannot edit" when you should have access
1. Check page-level permissions
2. Check parent page restrictions
3. Verify workspace membership
4. Clear browser cache

### Guest cannot access shared page
1. Ensure invite was accepted
2. Check guest hasn't exceeded workspace limit
3. Verify sharing is enabled (admin setting)

## Enterprise Features

- Domain-restricted sharing
- Permission groups
- Audit logs for sharing changes
- Default permission policies
