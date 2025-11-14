# Memory Intelligence Service - Render Deployment Guide

## Overview

This guide will help you deploy the Memory Intelligence Service (Python FastAPI) to Render.com for production use with your WordPress Memory Platform.

**Service Type:** Web Service
**Runtime:** Python 3.11
**Repository:** https://github.com/cognizonline/cogniz-memory-intelligence
**Estimated Cost:** Free tier initially, $7/month for production

---

## Prerequisites

- GitHub account with access to `cognizonline/cogniz-memory-intelligence`
- Render.com account (free signup at https://render.com)
- GitHub Personal Access Token (with repo access)

---

## Step 1: Push Code to GitHub

### 1.1 Initialize Git Repository (if not already done)

```bash
cd "C:\Users\Savannah Babette\mcp_ultra\mem0_aisl_16_09\Direct-MVP\memory-intelligence-service"
git init
```

### 1.2 Configure Git Remote

```bash
git remote add origin https://github.com/cognizonline/cogniz-memory-intelligence.git
```

### 1.3 Add All Files

```bash
git add .
git commit -m "Complete Intelligence Service implementation with WordPress integration"
```

### 1.4 Push to GitHub

```bash
# Use your GitHub token for authentication
git push -u origin main
```

**Authentication:** When prompted for username, enter your GitHub username. For password, use your GitHub Personal Access Token.

---

## Step 2: Create Render Web Service

### 2.1 Login to Render

1. Go to https://dashboard.render.com
2. Click **"New +"** button in top right
3. Select **"Web Service"**

### 2.2 Connect GitHub Repository

1. Click **"Connect account"** under GitHub
2. Authorize Render to access your GitHub
3. Search for `cogniz-memory-intelligence`
4. Click **"Connect"** next to the repository

### 2.3 Configure Web Service

Fill in the following settings:

| Field | Value |
|-------|-------|
| **Name** | `cogniz-memory-intelligence` |
| **Region** | Choose closest to your users (e.g., Oregon USA, Frankfurt EU) |
| **Branch** | `main` |
| **Root Directory** | Leave blank |
| **Runtime** | `Python 3` |
| **Build Command** | `pip install -r requirements.txt` |
| **Start Command** | `uvicorn app.main:app --host 0.0.0.0 --port $PORT` |
| **Instance Type** | `Free` (for testing) or `Starter` ($7/month for production) |

### 2.4 Advanced Settings

Click **"Advanced"** and configure:

#### Environment Variables

Add these environment variables (click "Add Environment Variable" for each):

| Key | Value | Notes |
|-----|-------|-------|
| `PYTHON_VERSION` | `3.11.0` | Pin Python version |
| `PORT` | `10000` | Render default port |
| `DB_HOST` | Leave empty initially | Optional: MySQL host if using external DB |
| `DB_USER` | Leave empty initially | Optional: MySQL username |
| `DB_PASSWORD` | Leave empty initially | Optional: MySQL password |
| `DB_NAME` | Leave empty initially | Optional: MySQL database name |

**Note:** The service works without database for duplicate detection and clustering. Database is only needed for consolidation features.

#### Health Check Path

- **Health Check Path:** `/health`
- **Health Check Interval:** `60` seconds

### 2.5 Create Web Service

1. Click **"Create Web Service"** at the bottom
2. Render will start deploying your service
3. Wait 3-5 minutes for initial deployment

---

## Step 3: Verify Deployment

### 3.1 Check Service Status

1. In Render dashboard, you'll see your service building
2. Wait for status to change from "Deploying" to "Live"
3. Look for the service URL (e.g., `https://cogniz-memory-intelligence.onrender.com`)

### 3.2 Test Health Endpoint

Open your browser or use curl:

```bash
curl https://cogniz-memory-intelligence.onrender.com/health
```

Expected response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "database": "connected"
}
```

**Note:** If database is not configured, you may see `"database": "error"` - this is OK for now.

### 3.3 Test Root Endpoint

```bash
curl https://cogniz-memory-intelligence.onrender.com/
```

Expected response:
```json
{
  "service": "Memory Intelligence Service",
  "version": "1.0.0",
  "status": "operational",
  "features": ["deduplication", "clustering", "priority_scoring"]
}
```

---

## Step 4: Configure WordPress Integration

### 4.1 Update Intelligence Settings in WordPress

1. Login to WordPress admin dashboard
2. Navigate to **Memory Platform > Dashboard**
3. Click **Intelligence** in sidebar
4. Click **Settings** tab
5. Update **Service URL** field:

```
https://cogniz-memory-intelligence.onrender.com
```

6. Enable **Duplicate Detection** and **Clustering**
7. Click **Save Settings**

### 4.2 Test WordPress Integration

1. Stay in Intelligence section
2. Click **"Analyze Now"** button
3. Wait 30-60 seconds
4. Check for insights appearing in the Insights tab

---

## Step 5: Production Optimization

### 5.1 Upgrade to Starter Instance

Free tier services on Render sleep after 15 minutes of inactivity. For production:

1. Go to Render dashboard
2. Click on your service
3. Click **"Settings"** tab
4. Scroll to **"Instance Type"**
5. Change from `Free` to `Starter` ($7/month)
6. Click **"Save Changes"**

**Benefits:**
- No sleep/cold starts
- 512MB RAM (vs 512MB shared)
- Better performance
- 24/7 availability

### 5.2 Enable Auto-Deploy

Render automatically deploys on every GitHub push by default. To verify:

1. Go to **Settings** tab
2. Scroll to **"Build & Deploy"**
3. Ensure **"Auto-Deploy"** is set to `Yes`

### 5.3 Configure Notifications

1. Go to **Settings** tab
2. Scroll to **"Notifications"**
3. Add your email for deployment notifications
4. Click **"Add Notification"**

---

## Step 6: Database Configuration (Optional)

If you want to use consolidation features and store cluster data, configure MySQL:

### 6.1 Create Render PostgreSQL Database

**Note:** Render doesn't offer MySQL. We'll use PostgreSQL instead and update code.

1. In Render dashboard, click **"New +"**
2. Select **"PostgreSQL"**
3. Configure:
   - **Name:** `cogniz-memory-db`
   - **Database:** `memory_platform`
   - **User:** Auto-generated
   - **Region:** Same as web service
   - **Instance Type:** `Free` or `Starter`
4. Click **"Create Database"**

### 6.2 Update Environment Variables

1. Go to your web service settings
2. Add environment variables from database:
   - `DB_HOST`: Internal Database URL (from DB info page)
   - `DB_USER`: Username (from DB info page)
   - `DB_PASSWORD`: Password (from DB info page)
   - `DB_NAME`: `memory_platform`
   - `DB_PORT`: `5432`

**Note:** If using PostgreSQL, you'll need to update `app/main.py` to use `psycopg2` instead of `mysql-connector-python`.

### 6.3 Alternative: Use WordPress MySQL Database

You can also configure the service to use your WordPress MySQL database:

1. Get database credentials from `wp-config.php`
2. Ensure database is accessible externally (check hosting provider)
3. Add environment variables:
   - `DB_HOST`: Your WordPress database host
   - `DB_USER`: WordPress database user
   - `DB_PASSWORD`: WordPress database password
   - `DB_NAME`: WordPress database name

**Warning:** Ensure proper security and backup before allowing external connections.

---

## Step 7: Monitoring and Maintenance

### 7.1 View Logs

1. In Render dashboard, click on your service
2. Click **"Logs"** tab
3. View real-time logs of requests and errors

### 7.2 Monitor Performance

1. Click **"Metrics"** tab
2. View:
   - Request volume
   - Response times
   - Memory usage
   - CPU usage

### 7.3 Set Up Alerts

1. Go to **Settings** > **Notifications**
2. Configure alerts for:
   - Service down
   - High error rate
   - High memory usage

---

## Step 8: Custom Domain (Optional)

To use a custom domain like `intelligence.cogniz.online`:

### 8.1 Add Custom Domain in Render

1. Go to **Settings** tab
2. Scroll to **"Custom Domains"**
3. Click **"Add Custom Domain"**
4. Enter: `intelligence.cogniz.online`
5. Render will provide DNS instructions

### 8.2 Configure DNS

1. Go to your DNS provider (Cloudflare, etc.)
2. Add CNAME record:
   - **Name:** `intelligence`
   - **Value:** `cogniz-memory-intelligence.onrender.com`
   - **TTL:** Auto or 3600

### 8.3 Enable SSL

Render automatically provisions SSL certificates via Let's Encrypt. Wait 5-10 minutes after DNS propagation.

### 8.4 Update WordPress

Update Service URL in WordPress to:
```
https://intelligence.cogniz.online
```

---

## Troubleshooting

### Service Won't Start

**Check logs:**
1. Go to Logs tab
2. Look for Python errors
3. Common issues:
   - Missing dependencies in `requirements.txt`
   - Port configuration errors
   - Import errors

**Fix:**
1. Update `requirements.txt` if dependencies missing
2. Ensure `Start Command` uses `$PORT` environment variable
3. Push changes to GitHub to trigger redeploy

### Health Check Failing

**Symptoms:**
- Service shows as "Unhealthy"
- Red status indicator

**Solutions:**
1. Verify `/health` endpoint works:
   ```bash
   curl https://your-service.onrender.com/health
   ```
2. Check logs for errors in health check
3. Ensure model can load (may need more memory - upgrade instance)

### Slow Response Times

**Causes:**
- Free tier sleeping (15 min inactivity)
- Model loading on first request
- Insufficient memory

**Solutions:**
1. Upgrade to Starter instance ($7/month)
2. Implement model caching
3. Use keep-alive pings

### Database Connection Errors

**If using external database:**
1. Verify credentials in environment variables
2. Check database allows external connections
3. Verify firewall/security group rules
4. Test connection from Render shell

### WordPress Can't Connect

**Symptoms:**
- "Intelligence Service unavailable" error
- Connection timeouts

**Solutions:**
1. Verify Service URL in WordPress settings
2. Check service is "Live" in Render
3. Test health endpoint manually
4. Check WordPress server can reach Render (firewall/proxy)

---

## API Endpoints Reference

For WordPress integration, the service provides:

### 1. Health Check
```
GET /health
Response: {"status": "healthy", "model_loaded": true}
```

### 2. Duplicate Detection
```
POST /api/v1/analyze/duplicates
Body: {
  "memories": [
    {"id": "mem_123", "content": "Sample memory content"},
    {"id": "mem_124", "content": "Another memory"}
  ],
  "threshold": 0.90
}
Response: {
  "duplicates": [
    {
      "memory_id": "mem_124",
      "duplicate_of": "mem_123",
      "similarity": 0.95
    }
  ]
}
```

### 3. Clustering
```
POST /api/v1/analyze/clusters
Body: {
  "memories": [
    {"id": "mem_123", "content": "Sample memory content"},
    {"id": "mem_124", "content": "Another memory"}
  ],
  "sensitivity": 0.70
}
Response: {
  "clusters": [
    {
      "cluster_id": "cluster_0",
      "memory_ids": ["mem_123", "mem_124"],
      "representative_id": "mem_123",
      "size": 2
    }
  ]
}
```

---

## Cost Breakdown

### Free Tier
- **Cost:** $0/month
- **RAM:** 512MB shared
- **Services:** 750 hours/month
- **Sleep:** After 15 min inactivity
- **Best for:** Development and testing

### Starter Tier (Recommended)
- **Cost:** $7/month
- **RAM:** 512MB
- **Services:** Always on
- **Sleep:** Never
- **Best for:** Production with low-medium traffic

### Standard Tier
- **Cost:** $25/month
- **RAM:** 2GB
- **Services:** Always on
- **Best for:** Production with high traffic

---

## Security Best Practices

### 1. Environment Variables
- Never commit sensitive credentials to GitHub
- Use Render's environment variable system
- Rotate credentials regularly

### 2. Database Access
- Use SSL/TLS for database connections
- Restrict database access to Render IPs only
- Use strong, unique passwords

### 3. API Security
- Consider adding API key authentication
- Implement rate limiting
- Monitor for suspicious activity

### 4. HTTPS Only
- Always use HTTPS URLs
- Enable HSTS headers
- Validate SSL certificates

---

## Next Steps

After successful deployment:

1. **Monitor Initial Usage**
   - Watch logs for errors
   - Check performance metrics
   - Verify WordPress integration working

2. **Optimize Performance**
   - Upgrade instance if needed
   - Configure caching
   - Monitor response times

3. **Enable Features**
   - Test duplicate detection
   - Test clustering
   - Enable background analysis in WordPress

4. **Scale as Needed**
   - Upgrade instance type based on usage
   - Consider database for consolidation features
   - Implement caching for frequently accessed data

---

## Support

- **Render Documentation:** https://render.com/docs
- **FastAPI Documentation:** https://fastapi.tiangolo.com
- **Issue Tracking:** GitHub Issues in your repository

---

## Deployment Checklist

- [ ] Code pushed to GitHub
- [ ] Render web service created
- [ ] Service deployed successfully
- [ ] Health check passing
- [ ] Service URL configured in WordPress
- [ ] Duplicate detection tested
- [ ] Clustering tested
- [ ] Background analysis working
- [ ] Monitoring configured
- [ ] Alerts set up
- [ ] Production instance tier selected
- [ ] Custom domain configured (optional)

---

**Deployment Date:** _______________
**Service URL:** _______________
**Deployed By:** _______________
