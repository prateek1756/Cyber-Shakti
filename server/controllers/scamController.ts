/**
 * Scam Alert Controller
 * 
 * Handles business logic for scam alert endpoints
 */

import { Request, Response } from 'express';
import { ScamReportModel, ScamReport } from '../models/ScamReport';
import { z } from 'zod';

// Validation schemas
const nearbyScamsSchema = z.object({
  latitude: z.number().min(-90).max(90),
  longitude: z.number().min(-180).max(180),
  radius: z.number().min(0.1).max(100).optional().default(10),
});

const createScamReportSchema = z.object({
  title: z.string().min(5).max(255),
  description: z.string().min(10).max(2000),
  scam_type: z.enum([
    'phishing',
    'phone_scam',
    'fake_website',
    'identity_theft',
    'investment_fraud',
    'romance_scam',
    'tech_support',
    'other'
  ]),
  latitude: z.number().min(-90).max(90),
  longitude: z.number().min(-180).max(180),
});

const updateStatusSchema = z.object({
  status: z.enum(['pending', 'verified', 'rejected']),
});

/**
 * Get nearby scam reports
 * GET /api/scams/nearby?latitude=X&longitude=Y&radius=Z
 */
export async function getNearbyScams(req: Request, res: Response): Promise<void> {
  try {
    // Parse and validate query parameters
    const { latitude, longitude, radius } = nearbyScamsSchema.parse({
      latitude: parseFloat(req.query.latitude as string),
      longitude: parseFloat(req.query.longitude as string),
      radius: req.query.radius ? parseFloat(req.query.radius as string) : 10,
    });

    // Fetch nearby scams from database
    const scams = await ScamReportModel.getNearby(latitude, longitude, radius);

    res.json({
      success: true,
      data: {
        location: { latitude, longitude },
        radius_km: radius,
        count: scams.length,
        scams: scams.map(scam => ({
          id: scam.id,
          title: scam.title,
          description: scam.description,
          scam_type: scam.scam_type,
          latitude: scam.latitude,
          longitude: scam.longitude,
          distance_km: parseFloat(scam.distance_km.toFixed(2)),
          created_at: scam.created_at,
        })),
      },
    });
  } catch (error) {
    if (error instanceof z.ZodError) {
      res.status(400).json({
        success: false,
        error: 'Invalid request parameters',
        details: error.errors,
      });
    } else {
      console.error('[ScamController] Error in getNearbyScams:', error);
      res.status(500).json({
        success: false,
        error: 'Failed to fetch nearby scams',
      });
    }
  }
}

/**
 * Create a new scam report
 * POST /api/scams/report
 */
export async function createScamReport(req: Request, res: Response): Promise<void> {
  try {
    // Validate request body
    const validatedData = createScamReportSchema.parse(req.body);

    // Get reporter IP address
    const reporterIp = req.ip || req.socket.remoteAddress || 'unknown';

    // Create scam report
    const report: ScamReport = {
      ...validatedData,
      reporter_ip: reporterIp,
      status: 'pending', // All new reports start as pending
    };

    const reportId = await ScamReportModel.create(report);

    res.status(201).json({
      success: true,
      data: {
        id: reportId,
        message: 'Scam report submitted successfully. It will be reviewed by our team.',
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
      console.error('[ScamController] Error in createScamReport:', error);
      res.status(500).json({
        success: false,
        error: 'Failed to create scam report',
      });
    }
  }
}

/**
 * Get scam report by ID
 * GET /api/scams/:id
 */
export async function getScamReport(req: Request, res: Response): Promise<void> {
  try {
    const id = parseInt(Array.isArray(req.params.id) ? req.params.id[0] : req.params.id, 10);

    if (isNaN(id)) {
      res.status(400).json({
        success: false,
        error: 'Invalid report ID',
      });
      return;
    }

    const scam = await ScamReportModel.getById(id);

    if (!scam) {
      res.status(404).json({
        success: false,
        error: 'Scam report not found',
      });
      return;
    }

    res.json({
      success: true,
      data: scam,
    });
  } catch (error) {
    console.error('[ScamController] Error in getScamReport:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to fetch scam report',
    });
  }
}

/**
 * Get all scam reports (with pagination)
 * GET /api/scams?limit=X&offset=Y&status=Z
 */
export async function getAllScamReports(req: Request, res: Response): Promise<void> {
  try {
    const limit = parseInt(req.query.limit as string) || 50;
    const offset = parseInt(req.query.offset as string) || 0;
    const status = req.query.status as 'pending' | 'verified' | 'rejected' | undefined;

    const scams = await ScamReportModel.getAll(limit, offset, status);

    res.json({
      success: true,
      data: {
        count: scams.length,
        limit,
        offset,
        scams,
      },
    });
  } catch (error) {
    console.error('[ScamController] Error in getAllScamReports:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to fetch scam reports',
    });
  }
}

/**
 * Update scam report status (admin only)
 * PATCH /api/scams/:id/status
 */
export async function updateScamStatus(req: Request, res: Response): Promise<void> {
  try {
    const id = parseInt(Array.isArray(req.params.id) ? req.params.id[0] : req.params.id, 10);

    if (isNaN(id)) {
      res.status(400).json({
        success: false,
        error: 'Invalid report ID',
      });
      return;
    }

    const { status } = updateStatusSchema.parse(req.body);

    const updated = await ScamReportModel.updateStatus(id, status);

    if (!updated) {
      res.status(404).json({
        success: false,
        error: 'Scam report not found',
      });
      return;
    }

    res.json({
      success: true,
      data: {
        id,
        status,
        message: 'Scam report status updated successfully',
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
      console.error('[ScamController] Error in updateScamStatus:', error);
      res.status(500).json({
        success: false,
        error: 'Failed to update scam report status',
      });
    }
  }
}

/**
 * Delete scam report (admin only)
 * DELETE /api/scams/:id
 */
export async function deleteScamReport(req: Request, res: Response): Promise<void> {
  try {
    const id = parseInt(Array.isArray(req.params.id) ? req.params.id[0] : req.params.id, 10);

    if (isNaN(id)) {
      res.status(400).json({
        success: false,
        error: 'Invalid report ID',
      });
      return;
    }

    const deleted = await ScamReportModel.delete(id);

    if (!deleted) {
      res.status(404).json({
        success: false,
        error: 'Scam report not found',
      });
      return;
    }

    res.json({
      success: true,
      data: {
        id,
        message: 'Scam report deleted successfully',
      },
    });
  } catch (error) {
    console.error('[ScamController] Error in deleteScamReport:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to delete scam report',
    });
  }
}

/**
 * Get scam statistics
 * GET /api/scams/stats
 */
export async function getScamStatistics(_req: Request, res: Response): Promise<void> {
  try {
    const stats = await ScamReportModel.getStatistics();

    res.json({
      success: true,
      data: stats,
    });
  } catch (error) {
    console.error('[ScamController] Error in getScamStatistics:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to fetch statistics',
    });
  }
}
