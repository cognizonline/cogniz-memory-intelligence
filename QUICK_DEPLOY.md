# Quick Deployment Guide

This is a condensed version of the deployment process. For full details, see [RENDER_DEPLOYMENT_GUIDE.md](RENDER_DEPLOYMENT_GUIDE.md).

## Step 1: Push to GitHub (5 minutes)

### 1.1 Navigate to Project

```bash
cd "C:\Users\Savannah Babette\mcp_ultra\mem0_aisl_16_09\Direct-MVP\memory-intelligence-service"
```

### 1.2 Initialize Git (if needed)

```bash
git init
git branch -M main
```

### 1.3 Configure GitHub Remote

```bash
git remote add origin https://github.com/cognizonline/cogniz-memory-intelligence.git
```

### 1.4 Commit and Push

```bash
git add .
git commit -m "Complete Intelligence Service with WordPress integration"
git push -u origin main
```

**Authentication:**
- Username: Your GitHub username
- Password: Use your GitHub Personal Access Token

If remote already exists:
```bash
git remote set-url origin https://github.com/cognizonline/cogniz-memory-intelligence.git
git push -u origin main --force
```

---

## Step 2: Deploy to Render (5 minutes)

### 2.1 Create Web Service

1. Go to https://dashboard.render.com
2. Click **New +** → **Web Service**
3. Connect GitHub repository: `cogniz-memory-intelligence`
4. Configure:
   - **Name:** `cogniz-memory-intelligence`
   - **Branch:** `main`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
   - **Instance Type:** `Free` (testing) or `Starter` (production $7/month)

### 2.2 Add Environment Variables

Click **Advanced** → **Add Environment Variable**:
- `PYTHON_VERSION` = `3.11.0`
- `PORT` = `10000`

### 2.3 Set Health Check

- **Health Check Path:** `/health`

### 2.4 Deploy

Click **Create Web Service**. Wait 3-5 minutes for deployment.

---

## Step 3: Test Deployment (2 minutes)

### 3.1 Get Service URL

Copy from Render dashboard (e.g., `https://cogniz-memory-intelligence.onrender.com`)

### 3.2 Test Health Endpoint

```bash
curl https://cogniz-memory-intelligence.onrender.com/health
```

Expected:
```json
{"status": "healthy", "model_loaded": true}
```

---

## Step 4: Configure WordPress (2 minutes)

1. Login to WordPress admin
2. Go to **Memory Platform > Dashboard > Intelligence**
3. Click **Settings** tab
4. Enter Service URL: `https://cogniz-memory-intelligence.onrender.com`
5. Enable features:
   - ✅ Duplicate Detection
   - ✅ Clustering
6. Click **Save Settings**
7. Click **Analyze Now** to test

---

## Troubleshooting

### Push Failed (Already Exists)

```bash
git remote set-url origin https://github.com/cognizonline/cogniz-memory-intelligence.git
git push -u origin main --force
```

### Render Build Failed

Check logs in Render dashboard. Common fixes:
- Verify `requirements.txt` has all dependencies
- Check `runtime.txt` has Python 3.11.0
- Ensure start command uses `$PORT`

### WordPress Can't Connect

1. Verify service is "Live" in Render
2. Test health endpoint manually
3. Check Service URL has no trailing slash
4. Ensure HTTPS (not HTTP)

---

## Production Checklist

- [ ] Code pushed to GitHub
- [ ] Render service deployed
- [ ] Health check passing
- [ ] Service URL in WordPress
- [ ] Test "Analyze Now" working
- [ ] Upgrade to Starter instance ($7/month)
- [ ] Enable auto-deploy
- [ ] Set up monitoring

---

**Total Time:** ~15 minutes
**Cost:** Free (testing) or $7/month (production)
