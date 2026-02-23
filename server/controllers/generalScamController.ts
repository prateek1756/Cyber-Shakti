/**
 * General Scam Report Controller
 * 
 * Handles business logic for general (non-location-based) scam reporting
 */

import { Request, Response } from 'express';
import { GeneralScamReportModel, GeneralScamReport } from '../models/GeneralScamReport';
import { z } from 'zod';

// Validation schemas
const createGeneralReportSchema = z.object({
  title: z.string().min(5).max(255),
  description: z.string().min(20).max(5000),
  scam_type: z.enum([
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
  ]),
  severity_level: z.enum(['low', 'medium', 'high', 'critical']),
  reporter_name: z.string().min(2).max(100).optional(),
  reporter_email: z.string().email().max(255).optional(),
  reporter_phone: z.string().max(20).optional(),
  evidence_url: z.string().url().max(500).optional(),
});

const updateStatusSchema = z.object({
  status: z.enum(['pending', 'approved', 'rejected', 'under_review']),
  admin_notes: z.string().max(2000).optional(),
  reviewed_by: z.number().int().positive().optional(),
});

/**
 * Create a new general scam report
 * POST /api/reports/submit
 */
export async function createGeneralReport(req: Request, res: Response): Promise<void> {
  try {
    // Validate request body
    const validatedData = createGeneralReportSchema.parse(req.body);

    // Get reporter IP address
    const reporterIp = req.ip || req.socket.remoteAddress || 'unknown';

    // Spam prevention: Check for duplicate reports from same IP within 5 minutes
    // TODO: Implement duplicate detection logic

    // Create report
    const report: GeneralScamReport = {
      ...validatedData,
      reporter_ip: reporterIp,
      status: 'pending',
    };

    const reportId = await GeneralScamReportModel.create(report);

    res.status(201).json({
      success: true,
      data: {
        id: reportId,
        message: 'Scam report submitted successfully. Our team will review it shortly.',
      },
    });
  } catch (error) {
    if (error instanceof z.ZodError) {
      res.status(400).json({
        success: false,
        error: 'Invalid request data',
        details: error.errors,
      });
    } else {
      console.error('[GeneralScamController] Error in createGeneralReport:', error);
      res.status(500).json({
        success: false,
        error: 'Failed to submit scam report',
      });
    }
  }
}

/**
 * Get report by ID
 * GET /api/reports/:id
 */
export async function getGeneralReport(req: Request, res: Response): Promise<void> {
  try {
    const id = parseInt(Array.isArray(req.params.id) ? req.params.id[0] : req.params.id, 10);

    if (isNaN(id)) {
      res.status(400).json({
        success: false,
        error: 'Invalid report ID',
      });
      return;
    }

    const report = await GeneralScamReportModel.getById(id);

    if (!report) {
      res.status(404).json({
        success: false,
        error: 'Report not found',
      });
      return;
    }

    // Get evidence files
    const evidence = await GeneralScamReportModel.getEvidence(id);

    res.json({
      success: true,
      data: {
        report,
        evidence,
      },
    });
  } catch (error) {
    console.error('[GeneralScamController] Error in getGeneralReport:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to fetch report',
    });
  }
}

/**
 * Get all reports with filtering and pagination
 * GET /api/reports?status=X&scam_type=Y&severity_level=Z&limit=N&offset=M
 */
export async function getAllGeneralReports(req: Request, res: Response): Promise<void> {
  try {
    const limit = parseInt(req.query.limit as string) || 50;
    const offset = parseInt(req.query.offset as string) || 0;
    
    const filters = {
      status: req.query.status as 'pending' | 'approved' | 'rejected' | 'under_review' | undefined,
      scam_type: req.query.scam_type as string | undefined,
      severity_level: req.query.severity_level as 'low' | 'medium' | 'high' | 'critical' | undefined,
    };

    const reports = await GeneralScamReportModel.getAll(limit, offset, filters);

    res.json({
      success: true,
      data: {
        count: reports.length,
        limit,
        offset,
        filters,
        reports,
      },
    });
  } catch (error) {
    console.error('[GeneralScamController] Error in getAllGeneralReports:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to fetch reports',
    });
  }
}

/**
 * Update report status (admin only)
 * PATCH /api/reports/:id/status
 */
export async function updateGeneralReportStatus(req: Request, res: Response): Promise<void> {
  try {
    const id = parseInt(Array.isArray(req.params.id) ? req.params.id[0] : req.params.id, 10);

    if (isNaN(id)) {
      res.status(400).json({
        success: false,
        error: 'Invalid report ID',
      });
      return;
    }

    const { status, admin_notes, reviewed_by } = updateStatusSchema.parse(req.body);

    const updated = await GeneralScamReportModel.updateStatus(
      id,
      status,
      admin_notes,
      reviewed_by
    );

    if (!updated) {
      res.status(404).json({
        success: false,
        error: 'Report not found',
      });
      return;
    }

    res.json({
      success: true,
      data: {
        id,
        status,
        message: 'Report status updated successfully',
      },
    });
  } catch (error) {
    if (error instanceof z.ZodError) {
      res.status(400).json({
        success: false,
        error: 'Invalid request data',
        details: error.errors,
      });
    } else {
      console.error('[GeneralScamController] Error in updateGeneralReportStatus:', error);
      res.status(500).json({
        success: false,
        error: 'Failed to update report status',
      });
    }
  }
}

/**
 * Delete report (admin only)
 * DELETE /api/reports/:id
 */
export async function deleteGeneralReport(req: Request, res: Response): Promise<void> {
  try {
    const id = parseInt(Array.isArray(req.params.id) ? req.params.id[0] : req.params.id, 10);

    if (isNaN(id)) {
      res.status(400).json({
        success: false,
        error: 'Invalid report ID',
      });
      return;
    }

    const deleted = await GeneralScamReportModel.delete(id);

    if (!deleted) {
      res.status(404).json({
        success: false,
        error: 'Report not found',
      });
      return;
    }

    res.json({
      success: true,
      data: {
        id,
        message: 'Report deleted successfully',
      },
    });
  } catch (error) {
    console.error('[GeneralScamController] Error in deleteGeneralReport:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to delete report',
    });
  }
}

/**
 * Get report statistics
 * GET /api/reports/stats
 */
export async function getGeneralReportStatistics(_req: Request, res: Response): Promise<void> {
  try {
    const stats = await GeneralScamReportModel.getStatistics();

    res.json({
      success: true,
      data: stats,
    });
  } catch (error) {
    console.error('[GeneralScamController] Error in getGeneralReportStatistics:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to fetch statistics',
    });
  }
}
