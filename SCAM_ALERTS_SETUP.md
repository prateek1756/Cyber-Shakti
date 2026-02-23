# Location-Based Scam Alert System - Setup Guide

## Overview
This guide will help you set up the Location-Based Scam Alert System for CyberShakti.

## Prerequisites
- Node.js (v18 or higher)
- MySQL (v8.0 or higher)
- npm or pnpm package manager

## Database Setup

### 1. Install MySQL
If you don't have MySQL installed:
- **Windows**: Download from [MySQL Downloads](https://dev.mysql.com/downloads/installer/)
- **Linux**: `sudo apt-get install mysql-server`
- **macOS**: `brew install mysql`

### 2. Create Database and Tables
Run the SQL schema file to create the database structure:

```bash
mysql -u root -p < database/schema.sql
```

Or manually:
1. Open MySQL command line: `mysql -u root -p`
2. Copy and paste the contents of `database/schema.sql`
3. Execute the commands

### 3. Verify Database Setup
```sql
USE cybershakti;
SHOW TABLES;
-- Should show: scam_reports, admin_users

DESCRIBE scam_reports;
-- Verify all columns are created

SELECT * FROM scam_reports;
-- Should show 3 sample records
```

## Environment Configuration

### 1. Create .env File
Copy the example environment file:
```bash
cp .env.example .env
```

### 2. Configure Database Connection
Edit `.env` and add your MySQL credentials:

```env
# Database Configuration
DB_HOST=localhost
DB_PORT=3306
DB_USER=root
DB_PASSWORD=your_mysql_password
DB_NAME=cybershakti

# Server Configuration
PORT=3000
NODE_ENV=development
```

## Install Dependencies

```bash
# Using npm
npm install

# Using pnpm (recommended)
pnpm install
```

This will install:
- `mysql2` - MySQL database driver
- `leaflet` - Interactive map library
- `zod` - Schema validation
- All other required dependencies

## Running the Application

### Development Mode
```bash
npm run dev
```

This starts:
- Frontend dev server on `http://localhost:5173`
- Backend API server on `http://localhost:3000`
- Python AI backend (if configured)

### Access the Scam Alerts Page
Navigate to: `http://localhost:5173/scam-alerts`

## Testing the System

### 1. Enable Location Access
- Click "Enable Location" button
- Allow browser location permission
- Your location will be displayed on the map

### 2. View Nearby Scams
- Sample scams from Delhi area should appear (if you're nearby)
- Adjust radius to see more/fewer results

### 3. Submit a Test Report
- Click "Report Scam" button
- Fill in the form:
  - Title: "Test Scam Report"
  - Type: Select any type
  - Description: "This is a test scam report"
- Submit the form
- Report will be created with "pending" status

### 4. Verify Database Entry
```sql
USE cybershakti;
SELECT * FROM scam_reports ORDER BY created_at DESC LIMIT 1;
```

## API Endpoints

### Public Endpoints
- `GET /api/scams/nearby?latitude=X&longitude=Y&radius=Z` - Get nearby scams
- `POST /api/scams/report` - Submit new scam report
- `GET /api/scams/stats` - Get statistics
- `GET /api/scams/:id` - Get specific report

### Admin Endpoints (TODO: Add authentication)
- `GET /api/scams` - Get all reports (paginated)
- `PATCH /api/scams/:id/status` - Update report status
- `DELETE /api/scams/:id` - Delete report

## Troubleshooting

### Database Connection Errors
```
Error: connect ECONNREFUSED 127.0.0.1:3306
```
**Solution**: 
- Verify MySQL is running: `sudo service mysql status`
- Check credentials in `.env` file
- Ensure database exists: `mysql -u root -p -e "SHOW DATABASES;"`

### Location Permission Denied
**Solution**:
- Browser must be on HTTPS or localhost
- Check browser location settings
- IP-based fallback (TODO: implement)

### Map Not Displaying
**Solution**:
- Check browser console for errors
- Verify Leaflet CSS is loaded
- Clear browser cache

### No Scams Showing
**Solution**:
- Verify sample data exists: `SELECT COUNT(*) FROM scam_reports;`
- Check if you're within radius of sample locations (Delhi area)
- Increase search radius to 50km or 100km

## Production Deployment

### 1. Build the Application
```bash
npm run build
```

### 2. Set Production Environment
```env
NODE_ENV=production
DB_HOST=your_production_db_host
```

### 3. Security Considerations
- [ ] Add authentication middleware for admin routes
- [ ] Implement rate limiting (already configured)
- [ ] Use HTTPS for production
- [ ] Sanitize user inputs (already using Zod validation)
- [ ] Add CORS configuration
- [ ] Set up database backups
- [ ] Monitor for SQL injection attempts

## Next Steps

### Recommended Enhancements
1. **IP-based Location Fallback**: Implement geolocation API for users who deny browser location
2. **Admin Dashboard**: Create admin panel for moderating reports
3. **Real-time Updates**: Add WebSocket support for live scam alerts
4. **Heatmap Visualization**: Show scam density on map
5. **Email Notifications**: Alert users about nearby scams
6. **Mobile App**: React Native version for mobile devices

## Support
For issues or questions, check:
- Database logs: `tail -f /var/log/mysql/error.log`
- Application logs: Check terminal output
- Browser console: F12 → Console tab
