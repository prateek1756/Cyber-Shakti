/**
 * General Scam Report Routes
 * 
 * API endpoints for general (non-location-based) scam reporting system
 */

import express from 'express';
import {
  createGeneralReport,
  getGeneralReport,
  getAllGeneralReports,
  updateGeneralReportStatus,
  deleteGeneralReport,
  getGeneralReportStatistics,
} from '../controllers/generalScamController';
import { rateLimiter } from '../middleware/rateLimiter';
import { upload, handleUploadError } from '../middleware/fileUpload';
import { GeneralScamReportModel } from '../models/GeneralScamReport';

const router = express.Router();

// ============================================
// Public Routes
// ============================================

/**
 * POST /api/reports/submit
 * Submit a new general scam report
 * 
 * Body (multipart/form-data):
 * - title: string (required)
 * - description: string (required)
 * - scam_type: enum (required)
 * - severity_level: enum (required)
 * - reporter_name: string (optional)
 * - reporter_email: string (optional)
 * - reporter_phone: string (optional)
 * - evidence: file[] (optional, max 5 files, 5MB each)
 */
router.post(
  '/submit',
  rateLimiter(60, 5), // Strict rate limit: 5 reports per hour
  upload.array('evidence', 5),
  handleUploadError,
  async (req, res) => {
    try {
      // Parse form data
      const reportData = {
        title: req.body.title,
        description: req.body.description,
        scam_type: req.body.scam_type,
        severity_level: req.body.severity_level,
        reporter_name: req.body.reporter_name,
        reporter_email: req.body.reporter_email,
        reporter_phone: req.body.reporter_phone,
      };

      // Get reporter IP
      const reporterIp = req.ip || req.socket.remoteAddress || 'unknown';

      // Create report
      const report = {
        ...reportData,
        reporter_ip: reporterIp,
        status: 'pending' as const,
      };

      const reportId = await GeneralScamReportModel.create(report);

      // Handle uploaded files
      const files = req.files as Express.Multer.File[];
      if (files && files.length > 0) {
        for (const file of files) {
          await GeneralScamReportModel.addEvidence({
            report_id: reportId,
            filename: file.filename,
            original_filename: file.originalname,
            file_path: file.path,
            file_size: file.size,
            mime_type: file.mimetype,
          });
        }
      }

      res.status(201).json({
        success: true,
        data: {
          id: reportId,
          message: 'Scam report submitted successfully. Our team will review it shortly.',
          files_uploaded: files?.length || 0,
        },
      });
    } catch (error) {
      console.error('[GeneralReports] Error in submit:', error);
      res.status(500).json({
        success: false,
        error: 'Failed to submit scam report',
      });
    }
  }
);

/**
 * GET /api/reports/stats
 * Get report statistics
 */
router.get('/stats', rateLimiter(60, 100), getGeneralReportStatistics);

/**
 * GET /api/reports/:id
 * Get a specific report by ID
 */
router.get('/:id', rateLimiter(60, 100), getGeneralReport);

// ============================================
// Admin Routes (TODO: Add authentication middleware)
// ============================================

/**
 * GET /api/reports
 * Get all reports with filtering and pagination
 * 
 * Query params:
 * - limit: number (optional, default: 50)
 * - offset: number (optional, default: 0)
 * - status: enum (optional)
 * - scam_type: enum (optional)
 * - severity_level: enum (optional)
 */
router.get('/', rateLimiter(60, 100), getAllGeneralReports);

/**
 * PATCH /api/reports/:id/status
 * Update report status (admin only)
 * 
 * Body:
 * - status: 'pending' | 'approved' | 'rejected' | 'under_review'
 * - admin_notes: string (optional)
 * - reviewed_by: number (optional)
 */
router.patch('/:id/status', rateLimiter(60, 50), updateGeneralReportStatus);

/**
 * DELETE /api/reports/:id
 * Delete a report (admin only)
 */
router.delete('/:id', rateLimiter(60, 20), deleteGeneralReport);

export default router;
