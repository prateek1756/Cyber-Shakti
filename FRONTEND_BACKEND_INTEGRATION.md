# Frontend-Backend Integration Complete

## ✅ Integration Status: COMPLETE

All frontend pages are now properly connected to backend APIs through the Express server.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                         FRONTEND                            │
│                    (React + Vite)                           │
│                   Port: 8080 (Vite)                         │
└─────────────────────────────────────────────────────────────┘
                            │
                            │ HTTP Requests
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    EXPRESS SERVER                           │
│                  (Node.js Backend)                          │
│                      Port: 8080                             │
│                                                             │
│  Routes:                                                    │
│  • /api/reports/*    → General Scam Reports                │
│  • /api/scams/*      → Location-Based Scams                │
│  • /api/fraud/*      → Fraud Detection (Proxy)             │
│  • /api/deepfake/*   → Deepfake Detection (Proxy)          │
│  • /api/scanner/*    → Phishing Scanner                    │
└─────────────────────────────────────────────────────────────┘
                            │
                ┌───────────┴───────────┐
                │                       │
                ▼                       ▼
┌──────────────────────┐    ┌──────────────────────┐
│  FRAUD DETECTION     │    │ DEEPFAKE DETECTION   │
│  (Python Flask)      │    │  (Python Flask)      │
│  Port: 8000          │    │  Port: 5001          │
└──────────────────────┘    └──────────────────────┘
```

---

## Integration Details

### 1. General Scam Reporting System ✅

**Frontend**: `client/pages/ReportScam.tsx`
**Backend**: `server/routes/generalReports.ts`
**Database**: `general_scam_reports` table

**Flow**:
```
User fills form → POST /api/reports/submit → Express validates → 
Multer handles files → MySQL stores data → Success response
```

**Features**:
- File upload (evidence)
- Form validation
- Rate limiting (5 reports/hour)
- SQL injection protection

---

### 2. Location-Based Scam Alerts ✅

**Frontend**: `client/pages/ScamAlerts.tsx`
**Backend**: `server/routes/scams.ts`
**Database**: `scam_reports` table

**Flow**:
```
User enables location → GET /api/scams/nearby?lat=X&lon=Y&radius=Z → 
Express queries MySQL → Returns nearby scams → Display on map
```

**Features**:
- Geolocation API
- Leaflet.js maps
- Haversine distance calculation
- Real-time filtering

---

### 3. Fraud Message Detection ✅ (NEWLY INTEGRATED)

**Frontend**: `client/pages/FraudDetection.tsx`
**Backend**: `server/routes/fraudDetection.ts` (Proxy)
**Service**: `python/fraud_detector.py` (Port 8000)

**Flow**:
```
User enters message → POST /api/fraud/detect → Express proxies to Python → 
AI analyzes message → Returns risk score → Display results
```

**Why Proxy?**
- Centralized API management
- Rate limiting
- Error handling
- Service health monitoring
- Easier deployment

**Before**:
```javascript
// Direct call to Python service
fetch("http://localhost:8000/detect", ...)
```

**After**:
```javascript
// Through Express proxy
fetch("/api/fraud/detect", ...)
```

---

### 4. Deepfake Detection ✅

**Frontend**: `client/pages/DeepfakeDetection.tsx`
**Backend**: `server/routes/deepfake-proxy.ts`
**Service**: `python/api_server.py` (Port 5001)

**Flow**:
```
User uploads image → POST /api/deepfake/detect → Express proxies to Python → 
AI model analyzes → Returns prediction → Display results
```

---

### 5. Phishing Scanner ✅

**Frontend**: `client/pages/PhishingScanner.tsx`
**Backend**: `server/routes/scanner.ts`

**Flow**:
```
User enters URL → POST /api/scanner/check → Express analyzes → 
Heuristic scoring → Returns risk assessment
```

---

## API Endpoints Summary

### General Scam Reports
```
POST   /api/reports/submit          - Submit new report
GET    /api/reports/:id             - Get report by ID
GET    /api/reports                 - List all reports (admin)
PATCH  /api/reports/:id/status      - Update status (admin)
DELETE /api/reports/:id             - Delete report (admin)
GET    /api/reports/stats           - Get statistics
```

### Location-Based Scams
```
GET    /api/scams/nearby            - Get nearby scams
POST   /api/scams/report            - Submit location-based report
GET    /api/scams/:id               - Get scam by ID
GET    /api/scams                   - List all scams (admin)
PATCH  /api/scams/:id/status        - Update status (admin)
DELETE /api/scams/:id               - Delete scam (admin)
GET    /api/scams/stats             - Get statistics
```

### Fraud Detection (NEW!)
```
POST   /api/fraud/detect            - Analyze message for fraud
GET    /api/fraud/health            - Check service health
```

### Deepfake Detection
```
POST   /api/deepfake/detect         - Analyze image
GET    /api/deepfake/stats          - Get statistics
```

### Phishing Scanner
```
POST   /api/scanner/check           - Check URL for phishing
```

---

## Environment Configuration

### .env Variables
```env
# Main Server
PORT=8080
NODE_ENV=development

# Database
DB_HOST=localhost
DB_PORT=3306
DB_USER=root
DB_PASSWORD=your_password
DB_NAME=cybershakti

# Python Services
FRAUD_API_URL=http://localhost:8000
FLASK_PORT=5001
FLASK_HOST=0.0.0.0
```

---

## Security Features

### 1. Rate Limiting
All endpoints have rate limiting:
- Report submission: 5 requests/hour
- Fraud detection: 30 requests/minute
- General queries: 100 requests/minute

### 2. Input Validation
- Zod schemas for type safety
- Length constraints
- Format validation
- SQL injection prevention

### 3. File Upload Security
- MIME type validation
- File size limits (5MB)
- Secure filename generation
- Isolated storage directory

### 4. CORS Configuration
- Configured for localhost development
- Restrictive in production

---

## Testing the Integration

### 1. Test General Scam Reporting
```bash
# Submit a report
curl -X POST http://localhost:8080/api/reports/submit \
  -F "title=Test Scam" \
  -F "description=This is a test scam report for testing purposes" \
  -F "scam_type=phishing" \
  -F "severity_level=medium"
```

### 2. Test Fraud Detection
```bash
# Analyze a message
curl -X POST http://localhost:8080/api/fraud/detect \
  -H "Content-Type: application/json" \
  -d '{"message": "URGENT! Click here to verify your account"}'
```

### 3. Test Location-Based Scams
```bash
# Get nearby scams
curl "http://localhost:8080/api/scams/nearby?latitude=28.6139&longitude=77.2090&radius=10"
```

### 4. Test Health Checks
```bash
# Check fraud detection service
curl http://localhost:8080/api/fraud/health

# Check deepfake service
curl http://localhost:8080/api/deepfake/stats
```

---

## Error Handling

### Frontend Error Handling
All pages implement:
- Try-catch blocks
- User-friendly error messages
- Toast notifications
- Loading states
- Retry mechanisms

### Backend Error Handling
All routes implement:
- Input validation
- Service availability checks
- Graceful degradation
- Detailed error logging
- Appropriate HTTP status codes

---

## Deployment Considerations

### Development
```bash
# Terminal 1: Main server (includes Express + Vite)
npm run dev

# Terminal 2: Fraud detection
python python/fraud_detector.py
```

### Production
1. **Build Frontend**:
   ```bash
   npm run build
   ```

2. **Start Express Server**:
   ```bash
   npm start
   ```

3. **Start Python Services**:
   ```bash
   # Fraud detection
   gunicorn -w 4 -b 0.0.0.0:8000 fraud_detector:app
   
   # Deepfake detection
   gunicorn -w 2 -b 0.0.0.0:5001 api_server:app
   ```

4. **Use Process Manager**:
   ```bash
   pm2 start ecosystem.config.js
   ```

---

## Files Modified/Created

### New Files
- `server/routes/fraudDetection.ts` - Fraud detection proxy
- `server/routes/generalReports.ts` - General scam reports
- `server/models/GeneralScamReport.ts` - Report model
- `server/controllers/generalScamController.ts` - Report controller
- `server/middleware/fileUpload.ts` - File upload handling
- `client/pages/ReportScam.tsx` - Report form page
- `python/fraud_detector.py` - Fraud detection service

### Modified Files
- `server/index.ts` - Added fraud detection routes
- `client/pages/FraudDetection.tsx` - Updated API endpoint
- `client/App.tsx` - Added report scam route
- `database/schema.sql` - Added general_scam_reports table
- `.env.example` - Added FRAUD_API_URL

---

## Benefits of This Architecture

### 1. Centralized API Management
- All API calls go through Express
- Consistent error handling
- Unified logging

### 2. Security
- Rate limiting at Express level
- Input validation before proxying
- Service isolation

### 3. Scalability
- Easy to add load balancing
- Service can be scaled independently
- Microservices architecture

### 4. Maintainability
- Clear separation of concerns
- Easy to update services
- Consistent API patterns

### 5. Monitoring
- Centralized health checks
- Service availability tracking
- Performance metrics

---

## Status Summary

✅ Frontend pages created and styled
✅ Backend routes implemented
✅ Database schema created
✅ File upload configured
✅ Rate limiting enabled
✅ Input validation implemented
✅ Error handling complete
✅ Proxy routes configured
✅ Python services integrated
✅ CORS configured
✅ Environment variables set
✅ Documentation complete

**All frontend and backend components are now properly integrated and working!**
