# Scam Reporting API Reference

## Base URL
```
http://localhost:8080/api/reports
```

## Endpoints

### 1. Submit Report
**POST** `/submit`

**Content-Type:** `multipart/form-data`

**Rate Limit:** 5 requests per hour per IP

**Request Body:**
```javascript
{
  title: string,              // Required, 5-255 chars
  description: string,        // Required, 20-5000 chars
  scam_type: enum,           // Required
  severity_level: enum,      // Required
  reporter_name: string,     // Optional, max 100 chars
  reporter_email: string,    // Optional, valid email
  reporter_phone: string,    // Optional, max 20 chars
  evidence: File[]           // Optional, max 5 files, 5MB each
}
```

**Scam Types:**
- `phishing`
- `phone_scam`
- `fake_website`
- `identity_theft`
- `investment_fraud`
- `romance_scam`
- `tech_support`
- `online_shopping`
- `job_scam`
- `lottery_scam`
- `charity_scam`
- `other`

**Severity Levels:**
- `low` - Minor inconvenience
- `medium` - Potential financial loss
- `high` - Significant financial loss
- `critical` - Severe financial loss

**Response:**
```json
{
  "success": true,
  "data": {
    "id": 123,
    "message": "Scam report submitted successfully...",
    "files_uploaded": 2
  }
}
```

**Example (JavaScript):**
```javascript
const formData = new FormData();
formData.append('title', 'Fake Amazon Email');
formData.append('description', 'Received phishing email...');
formData.append('scam_type', 'phishing');
formData.append('severity_level', 'medium');
formData.append('evidence', fileInput.files[0]);

const response = await fetch('/api/reports/submit', {
  method: 'POST',
  body: formData
});
```

---

### 2. Get Report by ID
**GET** `/:id`

**Response:**
```json
{
  "success": true,
  "data": {
    "report": {
      "id": 123,
      "title": "...",
      "description": "...",
      "scam_type": "phishing",
      "severity_level": "high",
      "status": "pending",
      "created_at": "2024-01-01T00:00:00Z"
    },
    "evidence": [
      {
        "id": 1,
        "filename": "...",
        "original_filename": "screenshot.png",
        "file_size": 102400,
        "mime_type": "image/png"
      }
    ]
  }
}
```

---

### 3. Get Statistics
**GET** `/stats`

**Response:**
```json
{
  "success": true,
  "data": {
    "total": 150,
    "pending": 25,
    "approved": 100,
    "rejected": 20,
    "under_review": 5,
    "byType": {
      "phishing": 45,
      "phone_scam": 30,
      "investment_fraud": 25
    },
    "bySeverity": {
      "low": 20,
      "medium": 60,
      "high": 50,
      "critical": 20
    }
  }
}
```

---

### 4. List All Reports (Admin)
**GET** `/`

**Query Parameters:**
- `status` - Filter by status (pending/approved/rejected/under_review)
- `scam_type` - Filter by scam type
- `severity_level` - Filter by severity
- `limit` - Number of results (default: 50)
- `offset` - Pagination offset (default: 0)

**Example:**
```
GET /api/reports?status=pending&severity_level=high&limit=20&offset=0
```

**Response:**
```json
{
  "success": true,
  "data": {
    "count": 20,
    "limit": 20,
    "offset": 0,
    "filters": {
      "status": "pending",
      "severity_level": "high"
    },
    "reports": [...]
  }
}
```

---

### 5. Update Report Status (Admin)
**PATCH** `/:id/status`

**Request Body:**
```json
{
  "status": "approved",
  "admin_notes": "Verified scam, approved for publication",
  "reviewed_by": 1
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "id": 123,
    "status": "approved",
    "message": "Report status updated successfully"
  }
}
```

---

### 6. Delete Report (Admin)
**DELETE** `/:id`

**Response:**
```json
{
  "success": true,
  "data": {
    "id": 123,
    "message": "Report deleted successfully"
  }
}
```

---

## Error Responses

### 400 Bad Request
```json
{
  "success": false,
  "error": "Invalid request data",
  "details": [
    {
      "path": ["title"],
      "message": "String must contain at least 5 character(s)"
    }
  ]
}
```

### 404 Not Found
```json
{
  "success": false,
  "error": "Report not found"
}
```

### 429 Too Many Requests
```json
{
  "success": false,
  "error": "Rate limit exceeded"
}
```

### 500 Internal Server Error
```json
{
  "success": false,
  "error": "Failed to submit scam report"
}
```

---

## File Upload Specifications

**Allowed Types:**
- Images: `image/jpeg`, `image/png`, `image/gif`, `image/webp`
- Documents: `application/pdf`, `text/plain`, `application/msword`, `application/vnd.openxmlformats-officedocument.wordprocessingml.document`

**Limits:**
- Max file size: 5MB per file
- Max files: 5 per report
- Total max: 25MB per report

**File Access:**
```
GET /uploads/evidence/{filename}
```

---

## Rate Limits

| Endpoint | Limit |
|----------|-------|
| POST /submit | 5 per hour |
| GET /:id | 100 per minute |
| GET /stats | 100 per minute |
| GET / (admin) | 100 per minute |
| PATCH /:id/status | 50 per minute |
| DELETE /:id | 20 per minute |

---

## Testing with cURL

### Submit Report
```bash
curl -X POST http://localhost:8080/api/reports/submit \
  -F "title=Test Scam Report" \
  -F "description=This is a detailed description of the scam..." \
  -F "scam_type=phishing" \
  -F "severity_level=medium" \
  -F "reporter_name=John Doe" \
  -F "reporter_email=john@example.com" \
  -F "evidence=@screenshot.png"
```

### Get Report
```bash
curl http://localhost:8080/api/reports/1
```

### Get Statistics
```bash
curl http://localhost:8080/api/reports/stats
```

### List Reports (Admin)
```bash
curl "http://localhost:8080/api/reports?status=pending&limit=10"
```

### Update Status (Admin)
```bash
curl -X PATCH http://localhost:8080/api/reports/1/status \
  -H "Content-Type: application/json" \
  -d '{"status":"approved","admin_notes":"Verified"}'
```

### Delete Report (Admin)
```bash
curl -X DELETE http://localhost:8080/api/reports/1
```
