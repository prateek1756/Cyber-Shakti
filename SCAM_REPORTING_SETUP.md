# User Scam Reporting System - Setup Guide

## Overview
Comprehensive scam reporting system with evidence uploads, severity levels, and admin moderation.

## Features
- General scam reporting (non-location-based)
- 12 scam type categories
- 4 severity levels
- Evidence file uploads
- Optional reporter contact info
- Admin moderation workflow
- Rate limiting and spam prevention

## Setup Instructions

### 1. Database Setup
The schema is already included in `database/schema.sql`. Run:
```bash
setup-database.bat
```

### 2. Install Dependencies
```bash
npm install
```

This installs `multer` for file uploads.

### 3. Create Upload Directory
The system automatically creates `uploads/evidence/` directory.

### 4. Run Application
```bash
npm run dev
```

### 5. Access Features
- Report Form: `http://localhost:5173/report-scam`
- API Docs: See below

## API Endpoints

### Public
- `POST /api/reports/submit` - Submit report with files
- `GET /api/reports/:id` - Get report details
- `GET /api/reports/stats` - Get statistics

### Admin (TODO: Add auth)
- `GET /api/reports` - List all reports (filtered)
- `PATCH /api/reports/:id/status` - Update status
- `DELETE /api/reports/:id` - Delete report

## File Upload
- Max 5 files per report
- 5MB per file
- Allowed: images, PDF, TXT, DOC, DOCX
- Stored in: `uploads/evidence/`

## Security
- Rate limiting: 5 reports/hour per IP
- SQL injection prevention (prepared statements)
- File type validation
- Input sanitization (Zod)

## Next Steps
1. Implement admin authentication
2. Add duplicate detection
3. Create admin dashboard
4. Add email notifications
