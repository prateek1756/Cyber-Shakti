-- ============================================
-- CyberShakti - Location-Based Scam Alert System
-- MySQL Database Schema
-- ============================================

-- Create database (if not exists)
CREATE DATABASE IF NOT EXISTS cybershakti CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE cybershakti;

-- ============================================
-- Scam Reports Table
-- ============================================
CREATE TABLE IF NOT EXISTS scam_reports (
  id INT AUTO_INCREMENT PRIMARY KEY,
  title VARCHAR(255) NOT NULL,
  description TEXT NOT NULL,
  scam_type ENUM(
    'phishing',
    'phone_scam',
    'fake_website',
    'identity_theft',
    'investment_fraud',
    'romance_scam',
    'tech_support',
    'other'
  ) NOT NULL DEFAULT 'other',
  latitude DOUBLE NOT NULL,
  longitude DOUBLE NOT NULL,
  location_point POINT NOT NULL,
  reporter_ip VARCHAR(45),
  status ENUM('pending', 'verified', 'rejected') NOT NULL DEFAULT 'pending',
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  
  -- Indexes for performance
  INDEX idx_created_at (created_at),
  INDEX idx_status (status),
  INDEX idx_scam_type (scam_type),
  
  -- Spatial index for geolocation queries
  SPATIAL INDEX idx_location_point (location_point)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ============================================
-- Trigger to auto-populate location_point from lat/lon
-- ============================================
DELIMITER $$

CREATE TRIGGER before_insert_scam_reports
BEFORE INSERT ON scam_reports
FOR EACH ROW
BEGIN
  SET NEW.location_point = POINT(NEW.longitude, NEW.latitude);
END$$

CREATE TRIGGER before_update_scam_reports
BEFORE UPDATE ON scam_reports
FOR EACH ROW
BEGIN
  IF NEW.latitude != OLD.latitude OR NEW.longitude != OLD.longitude THEN
    SET NEW.location_point = POINT(NEW.longitude, NEW.latitude);
  END IF;
END$$

DELIMITER ;

-- ============================================
-- Sample Data (for testing)
-- ============================================
INSERT INTO scam_reports (title, description, scam_type, latitude, longitude) VALUES
('Fake Parcel Delivery SMS', 'Received SMS claiming failed delivery with suspicious link', 'phishing', 28.6139, 77.2090),
('Tech Support Scam Call', 'Caller claimed to be from Microsoft, asked for remote access', 'tech_support', 28.6200, 77.2100),
('Investment Fraud Website', 'Fake cryptocurrency investment platform promising high returns', 'investment_fraud', 28.6150, 77.2050);

-- ============================================
-- Stored Procedure: Get Nearby Scams
-- ============================================
DELIMITER $$

CREATE PROCEDURE GetNearbyScams(
  IN user_lat DOUBLE,
  IN user_lon DOUBLE,
  IN radius_km DOUBLE
)
BEGIN
  -- Calculate distance using Haversine formula
  -- Returns scams within specified radius, sorted by distance
  SELECT 
    id,
    title,
    description,
    scam_type,
    latitude,
    longitude,
    created_at,
    status,
    (
      6371 * ACOS(
        COS(RADIANS(user_lat)) * 
        COS(RADIANS(latitude)) * 
        COS(RADIANS(longitude) - RADIANS(user_lon)) + 
        SIN(RADIANS(user_lat)) * 
        SIN(RADIANS(latitude))
      )
    ) AS distance_km
  FROM scam_reports
  WHERE status = 'verified'
  HAVING distance_km <= radius_km
  ORDER BY distance_km ASC
  LIMIT 50;
END$$

DELIMITER ;

-- ============================================
-- View: Recent Verified Scams
-- ============================================
CREATE OR REPLACE VIEW recent_verified_scams AS
SELECT 
  id,
  title,
  description,
  scam_type,
  latitude,
  longitude,
  created_at,
  DATEDIFF(NOW(), created_at) AS days_ago
FROM scam_reports
WHERE status = 'verified'
  AND created_at >= DATE_SUB(NOW(), INTERVAL 30 DAY)
ORDER BY created_at DESC;

-- ============================================
-- Admin User (for moderation)
-- ============================================
CREATE TABLE IF NOT EXISTS admin_users (
  id INT AUTO_INCREMENT PRIMARY KEY,
  username VARCHAR(50) UNIQUE NOT NULL,
  password_hash VARCHAR(255) NOT NULL,
  email VARCHAR(255) UNIQUE NOT NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ============================================
-- General Scam Reports Table (Non-Location Based)
-- ============================================
CREATE TABLE IF NOT EXISTS general_scam_reports (
  id INT AUTO_INCREMENT PRIMARY KEY,
  title VARCHAR(255) NOT NULL,
  description TEXT NOT NULL,
  scam_type ENUM(
    'phishing',
    'phone_scam',
    'fake_website',
    'identity_theft',
    'investment_fraud',
    'romance_scam',
    'tech_support',
    'online_shopping',
    'job_scam',
    'lottery_scam',
    'charity_scam',
    'other'
  ) NOT NULL DEFAULT 'other',
  severity_level ENUM('low', 'medium', 'high', 'critical') NOT NULL DEFAULT 'medium',
  reporter_name VARCHAR(100),
  reporter_email VARCHAR(255),
  reporter_phone VARCHAR(20),
  evidence_url VARCHAR(500),
  evidence_filename VARCHAR(255),
  reporter_ip VARCHAR(45),
  status ENUM('pending', 'approved', 'rejected', 'under_review') NOT NULL DEFAULT 'pending',
  admin_notes TEXT,
  reviewed_by INT,
  reviewed_at TIMESTAMP NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  
  -- Indexes for performance
  INDEX idx_created_at (created_at),
  INDEX idx_status (status),
  INDEX idx_scam_type (scam_type),
  INDEX idx_severity_level (severity_level),
  INDEX idx_reporter_email (reporter_email),
  
  -- Foreign key for admin reviewer
  FOREIGN KEY (reviewed_by) REFERENCES admin_users(id) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ============================================
-- Evidence Files Table (for file upload tracking)
-- ============================================
CREATE TABLE IF NOT EXISTS scam_evidence_files (
  id INT AUTO_INCREMENT PRIMARY KEY,
  report_id INT NOT NULL,
  filename VARCHAR(255) NOT NULL,
  original_filename VARCHAR(255) NOT NULL,
  file_path VARCHAR(500) NOT NULL,
  file_size INT NOT NULL,
  mime_type VARCHAR(100) NOT NULL,
  uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  
  -- Foreign key
  FOREIGN KEY (report_id) REFERENCES general_scam_reports(id) ON DELETE CASCADE,
  
  -- Index
  INDEX idx_report_id (report_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ============================================
-- View: Pending Reports for Admin Review
-- ============================================
CREATE OR REPLACE VIEW pending_reports_view AS
SELECT 
  id,
  title,
  scam_type,
  severity_level,
  reporter_name,
  reporter_email,
  status,
  created_at,
  DATEDIFF(NOW(), created_at) AS days_pending
FROM general_scam_reports
WHERE status = 'pending'
ORDER BY severity_level DESC, created_at ASC;

-- ============================================
-- View: Report Statistics by Type and Severity
-- ============================================
CREATE OR REPLACE VIEW report_statistics_view AS
SELECT 
  scam_type,
  severity_level,
  COUNT(*) as report_count,
  SUM(CASE WHEN status = 'approved' THEN 1 ELSE 0 END) as approved_count,
  SUM(CASE WHEN status = 'rejected' THEN 1 ELSE 0 END) as rejected_count,
  SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END) as pending_count
FROM general_scam_reports
GROUP BY scam_type, severity_level;

-- ============================================
-- Sample Data for General Reports
-- ============================================
INSERT INTO general_scam_reports (title, description, scam_type, severity_level, reporter_name, reporter_email) VALUES
('Fake Amazon Customer Service', 'Received call claiming to be Amazon, asking for account verification', 'phone_scam', 'high', 'John Doe', 'john@example.com'),
('Phishing Email from "PayPal"', 'Email asking to verify payment information with suspicious link', 'phishing', 'medium', 'Jane Smith', 'jane@example.com'),
('Fake Job Posting', 'Job offer requiring upfront payment for training materials', 'job_scam', 'high', NULL, NULL);
