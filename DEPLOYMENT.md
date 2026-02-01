# CyberShakti Deployment Guide

## Quick Deploy Links

### Vercel
[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/YOUR_USERNAME/CyberShakti)

### Netlify
[![Deploy to Netlify](https://www.netlify.com/img/deploy/button.svg)](https://app.netlify.com/start/deploy?repository=https://github.com/YOUR_USERNAME/CyberShakti)

## Manual Deployment Steps

### For Vercel:

1. **Connect Repository**
   - Go to [vercel.com](https://vercel.com)
   - Click "New Project"
   - Import your GitHub repository

2. **Configure Build Settings** (Auto-detected from vercel.json)
   - Build Command: `pnpm build:client`
   - Output Directory: `dist/spa`
   - Install Command: `pnpm install`

3. **Deploy**
   - Click "Deploy"
   - Your app will be live at `https://your-project.vercel.app`

### For Netlify:

1. **Connect Repository**
   - Go to [netlify.com](https://netlify.com)
   - Click "New site from Git"
   - Choose your GitHub repository

2. **Configure Build Settings** (Auto-detected from netlify.toml)
   - Build Command: `pnpm build:client`
   - Publish Directory: `dist/spa`

3. **Deploy**
   - Click "Deploy site"
   - Your app will be live at `https://your-project.netlify.app`

## Environment Variables

If you have any environment variables in `.env`, add them in your deployment platform:

**Vercel**: Project Settings → Environment Variables
**Netlify**: Site Settings → Environment Variables

## Custom Domain

Both platforms support custom domains:
- **Vercel**: Project Settings → Domains
- **Netlify**: Site Settings → Domain Management

## Build Troubleshooting

If build fails:
1. Check Node.js version (should be 18+)
2. Ensure `pnpm` is available (both platforms support it)
3. Check build logs for specific errors

## Notes

- The app is configured as a Single Page Application (SPA)
- Client-side routing is handled by React Router
- All routes redirect to `index.html` for proper SPA behavior
- The Python AI features are not included in the web deployment