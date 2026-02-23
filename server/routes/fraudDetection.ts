/**
 * Fraud Detection Routes
 * 
 * Proxy routes for fraud detection API
 */

import express, { Request, Response } from 'express';
import { rateLimiter } from '../middleware/rateLimiter';

const router = express.Router();

// Fraud detection service URL
const FRAUD_API_URL = process.env.FRAUD_API_URL || 'http://127.0.0.1:8000';

/**
 * POST /api/fraud/detect
 * Analyze message for fraud indicators
 * 
 * Body:
 * - message: string (required)
 */
// router.post('/detect', rateLimiter(60, 30), async (req: Request, res: Response) => {
router.post('/detect', async (req: Request, res: Response) => {
  try {
    const { message } = req.body;

    // In Vercel, we should be hitting the Python shim directly from the frontend.
    // However, if we hit this Express route, we should return a meaningful error or redirect.
    if (process.env.VERCEL) {
      return res.status(404).json({
        success: false,
        error: 'This endpoint is handled by Vercel serverless functions directly at /api/fraud/detect'
      });
    }

    console.log('[FraudDetection] Analyzing message:', message);
    console.log('[FraudDetection] Using API URL:', FRAUD_API_URL);

    if (!message || typeof message !== 'string') {
      res.status(400).json({
        success: false,
        error: 'Message is required and must be a string',
      });
      return;
    }

    if (message.trim().length === 0) {
      res.status(400).json({
        success: false,
        error: 'Message cannot be empty',
      });
      return;
    }

    // Forward request to fraud detection service
    const response = await fetch(`${FRAUD_API_URL}/detect`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ message }),
    });

    if (!response.ok) {
      throw new Error(`Fraud API returned status ${response.status}`);
    }

    const data = await response.json();

    res.json({
      success: true,
      data,
    });
  } catch (error) {
    console.error('[FraudDetection] Error:', error);

    // Check if fraud detection service is down
    if (error instanceof Error && error.message.includes('fetch')) {
      res.status(503).json({
        success: false,
        error: 'Fraud detection service is currently unavailable. Please try again later.',
      });
      return;
    }

    res.status(500).json({
      success: false,
      error: 'Failed to analyze message',
    });
  }
});

/**
 * GET /api/fraud/health
 * Check fraud detection service health
 */
router.get('/health', async (_req: Request, res: Response) => {
  try {
    const response = await fetch(`${FRAUD_API_URL}/health`);

    if (!response.ok) {
      throw new Error('Service unhealthy');
    }

    const data = await response.json();

    res.json({
      success: true,
      data,
    });
  } catch (error) {
    res.status(503).json({
      success: false,
      error: 'Fraud detection service is unavailable',
    });
  }
});

export default router;
