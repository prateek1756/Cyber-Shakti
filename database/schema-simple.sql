-- ============================================
-- CyberShakti - Database Schema (Simplified)
-- ============================================

CREATE DATABASE IF NOT EXISTS cybershakti CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE cybershakti;

-- ============================================
-- Admin Users Table
-- ============================================
CREATE TABLE IF NOT EXISTS admin_users (
  id INT AUTO_INCREMENT PRIMARY KEY,
  username VARCHAR(50) UNIQUE NOT NULL,
  password_hash VARCHAR(255) NOT NULL,
  email VARCHAR(255) UNIQUE NOT NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ============================================
-- Location-Based Scam Reports Table
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
  
  INDEX idx_created_at (created_at),
  INDEX idx_status (status),
  INDEX idx_scam_type (scam_type),
  SPATIAL INDEX idx_location_point (location_point)
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
  
  INDEX idx_created_at (created_at),
  INDEX idx_status (status),
  INDEX idx_scam_type (scam_type),
  INDEX idx_severity_level (severity_level),
  INDEX idx_reporter_email (reporter_email),
  
  FOREIGN KEY (reviewed_by) REFERENCES admin_users(id) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ============================================
-- Evidence Files Table
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
  
  FOREIGN KEY (report_id) REFERENCES general_scam_reports(id) ON DELETE CASCADE,
  INDEX idx_report_id (report_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ============================================
-- Sample Data
-- ============================================
INSERT INTO general_scam_reports (title, description, scam_type, severity_level, reporter_name, reporter_email) VALUES
('Fake Amazon Customer Service', 'Received call claiming to be Amazon, asking for account verification', 'phone_scam', 'high', 'John Doe', 'john@example.com'),
('Phishing Email from "PayPal"', 'Email asking to verify payment information with suspicious link', 'phishing', 'medium', 'Jane Smith', 'jane@example.com'),
('Fake Job Posting', 'Job offer requiring upfront payment for training materials', 'job_scam', 'high', NULL, NULL);

INSERT INTO scam_reports (title, description, scam_type, latitude, longitude, location_point) VALUES
('Fake Parcel Delivery SMS', 'Received SMS claiming failed delivery with suspicious link', 'phishing', 28.6139, 77.2090, POINT(77.2090, 28.6139)),
('Tech Support Scam Call', 'Caller claimed to be from Microsoft, asked for remote access', 'tech_support', 28.6200, 77.2100, POINT(77.2100, 28.6200)),
('Investment Fraud Website', 'Fake cryptocurrency investment platform promising high returns', 'investment_fraud', 28.6150, 77.2050, POINT(77.2050, 28.6150));
