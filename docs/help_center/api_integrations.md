---
doc_id: help_api_integrations
title: API and Integrations Guide
version: v1
tags: [import_export_sync, feature_request]
source_url: https://help.example.com/integrations/api
---

# API and Integrations

## Available Integrations

### Built-in
- Google Drive
- Dropbox
- Figma
- Slack
- GitHub
- Jira

### Via Zapier/Make
- 2000+ apps
- Automated workflows
- No code required

## Setting Up Integrations

### Google Drive
1. Settings → Integrations
2. Click Connect next to Google Drive
3. Authorize access
4. Use /google-drive in pages

### Slack
1. Settings → Integrations → Slack
2. Choose workspace to connect
3. Select channels for notifications
4. Configure notification triggers

## API Access

### Getting Started
1. Settings → Developer
2. Create new integration
3. Copy API token
4. Review documentation

### API Capabilities
- Read/write pages
- Manage databases
- Search content
- User management (admin)

### Rate Limits
- 3 requests/second (standard)
- 10 requests/second (business)
- Custom limits (enterprise)

### Authentication
- Bearer token authentication
- Tokens never expire (revoke manually)
- Scopes limit access

## Webhooks

### Supported Events
- Page updated
- Database row added
- Comment added
- Member joined

### Setup
1. Settings → Developer → Webhooks
2. Add endpoint URL
3. Select events
4. Test connection

## Common Issues

### Integration not working
1. Re-authorize the connection
2. Check token permissions
3. Verify rate limits
4. Review error logs

### Data sync delayed
- Normal delay: up to 5 minutes
- Check integration status page
- Retry manual sync
