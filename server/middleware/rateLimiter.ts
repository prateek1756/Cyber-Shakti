/**
 * Rate Limiting Middleware
 * 
 * Prevents abuse by limiting the number of requests per IP address
 */

import { Request, Response, NextFunction } from 'express';

// Store for tracking request counts per IP
const requestCounts = new Map<string, { count: number; resetTime: number }>();

/**
 * Rate limiter middleware factory
 * 
 * @param windowSeconds - Time window in seconds
 * @param maxRequests - Maximum requests allowed in the time window
 * @returns Express middleware function
 */
export function rateLimiter(windowSeconds: number, maxRequests: number) {
  return (req: Request, res: Response, next: NextFunction): void => {
    const ip = req.ip || req.socket.remoteAddress || 'unknown';
    const now = Date.now();
    const windowMs = windowSeconds * 1000;

    // Get or create request count for this IP
    let record = requestCounts.get(ip);

    if (!record || now > record.resetTime) {
      // Create new record or reset expired one
      record = {
        count: 0,
        resetTime: now + windowMs,
      };
      requestCounts.set(ip, record);
    }

    // Increment request count
    record.count++;

    // Check if limit exceeded
    if (record.count > maxRequests) {
      const retryAfter = Math.ceil((record.resetTime - now) / 1000);
      
      res.status(429).json({
        success: false,
        error: 'Too many requests',
        message: `Rate limit exceeded. Please try again in ${retryAfter} seconds.`,
        retryAfter,
      });
      return;
    }

    // Add rate limit headers
    res.setHeader('X-RateLimit-Limit', maxRequests.toString());
    res.setHeader('X-RateLimit-Remaining', (maxRequests - record.count).toString());
    res.setHeader('X-RateLimit-Reset', new Date(record.resetTime).toISOString());

    next();
  };
}

/**
 * Clean up expired records periodically
 * Should be called on application startup
 */
export function startRateLimiterCleanup(): void {
  setInterval(() => {
    const now = Date.now();
    for (const [ip, record] of requestCounts.entries()) {
      if (now > record.resetTime) {
        requestCounts.delete(ip);
      }
    }
  }, 60000); // Clean up every minute
}
