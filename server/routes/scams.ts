/**
 * Scam Alert Routes
 * 
 * API endpoints for location-based scam alert system
 */

import express from 'express';
import {
  getNearbyScams,
  createScamReport,
  getScamReport,
  getAllScamReports,
  updateScamStatus,
  deleteScamReport,
  getScamStatistics,
} from '../controllers/scamController';
import { rateLimiter } from '../middleware/rateLimiter';

const router = express.Router();

// ============================================
// Public Routes
// ============================================

/**
 * GET /api/scams/nearby
 * Get scam reports near a specific location
 * 
 * Query params:
 * - latitude: number (required)
 * - longitude: number (required)
 * - radius: number (optional, default: 10 km)
 */
router.get('/nearby', rateLimiter(60, 100), getNearbyScams);

/**
 * POST /api/scams/report
 * Submit a new scam report
 * 
 * Body:
 * - title: string (required)
 * - description: string (required)
 * - scam_type: enum (required)
 * - latitude: number (required)
 * - longitude: number (required)
 */
router.post('/report', rateLimiter(60, 10), createScamReport);

/**
 * GET /api/scams/stats
 * Get scam statistics
 */
router.get('/stats', rateLimiter(60, 100), getScamStatistics);

/**
 * GET /api/scams/:id
 * Get a specific scam report by ID
 */
router.get('/:id', rateLimiter(60, 100), getScamReport);

// ============================================
// Admin Routes (TODO: Add authentication middleware)
// ============================================

/**
 * GET /api/scams
 * Get all scam reports (with pagination)
 * 
 * Query params:
 * - limit: number (optional, default: 50)
 * - offset: number (optional, default: 0)
 * - status: enum (optional)
 */
router.get('/', rateLimiter(60, 100), getAllScamReports);

/**
 * PATCH /api/scams/:id/status
 * Update scam report status (admin only)
 * 
 * Body:
 * - status: 'pending' | 'verified' | 'rejected'
 */
router.patch('/:id/status', rateLimiter(60, 50), updateScamStatus);

/**
 * DELETE /api/scams/:id
 * Delete a scam report (admin only)
 */
router.delete('/:id', rateLimiter(60, 20), deleteScamReport);

export default router;
