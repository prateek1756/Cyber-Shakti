/**
 * Flask Proxy Middleware
 * 
 * This module provides middleware to proxy requests from Express to the Flask
 * server. It handles Flask unavailability gracefully and preserves request/response
 * headers and body.
 * 
 * Validates: Requirements 3.5, 9.3
 */

import { Request, Response } from 'express';
import { PythonBridge } from '../python-bridge';

/**
 * Options for the proxy middleware
 */
export interface ProxyMiddlewareOptions {
  /** PythonBridge instance to check Flask readiness */
  bridge: PythonBridge;
  /** Target Flask server URL */
  targetUrl: string;
}

/**
 * Error response when Flask is unavailable
 */
interface FlaskUnavailableError {
  error: string;
  message: string;
  suggestion: string;
  timestamp: string;
}

/**
 * Create a proxy middleware for deepfake endpoints
 * 
 * This middleware:
 * 1. Checks if Flask is ready before proxying
 * 2. Returns 503 error if Flask is unavailable
 * 3. Forwards requests to Flask if ready
 * 4. Preserves request/response headers and body
 * 
 * @param options - Proxy middleware options
 * @returns Express middleware function
 */
export function createDeepfakeProxy(options: ProxyMiddlewareOptions) {
  const { bridge, targetUrl } = options;

  return async (req: Request, res: Response) => {
    // Check if Flask is ready
    if (!bridge.isFlaskReady()) {
      // Flask is unavailable, return 503 error
      const errorResponse: FlaskUnavailableError = {
        error: 'Flask server unavailable',
        message: 'The deepfake detection service is currently unavailable',
        suggestion: 'Restart the development server with: pnpm dev',
        timestamp: new Date().toISOString(),
      };

      return res.status(503).json(errorResponse);
    }

    try {
      // Build the target URL for Flask
      // req.path will be like "/stats" when mounted at "/api/deepfake"
      // We need to reconstruct the full path: /api/deepfake/stats
      const flaskUrl = `${targetUrl}/api/deepfake${req.path}`;
      
      // Prepare query string if present
      const queryString = new URLSearchParams(req.query as Record<string, string>).toString();
      const fullUrl = queryString ? `${flaskUrl}?${queryString}` : flaskUrl;

      // Prepare headers to forward
      const headers: Record<string, string> = {};
      
      // Copy relevant headers (exclude host-related headers)
      Object.keys(req.headers).forEach((key) => {
        const lowerKey = key.toLowerCase();
        // Skip host-related headers that should not be forwarded
        if (lowerKey !== 'host' && lowerKey !== 'connection') {
          const value = req.headers[key];
          if (typeof value === 'string') {
            headers[key] = value;
          } else if (Array.isArray(value)) {
            headers[key] = value.join(', ');
          }
        }
      });

      // Prepare request body
      let body: BodyInit | undefined;
      
      if (req.method !== 'GET' && req.method !== 'HEAD') {
        // For POST, PUT, PATCH, etc., we need to send the body
        if (req.is('application/json')) {
          body = JSON.stringify(req.body);
        } else if (req.is('multipart/form-data')) {
          // For multipart/form-data, we need to handle it specially
          // Express with multer or body-parser should have already parsed it
          // We'll need to reconstruct it or pass the raw body
          // For now, we'll use the raw body if available
          body = (req as any).rawBody || JSON.stringify(req.body);
        } else {
          // For other content types, try to use the body as-is
          body = JSON.stringify(req.body);
        }
      }

      // Make the request to Flask
      const fetchOptions: RequestInit = {
        method: req.method,
        headers,
      };

      if (body !== undefined) {
        fetchOptions.body = body;
      }

      const flaskResponse = await fetch(fullUrl, fetchOptions);

      // Forward response status code
      res.status(flaskResponse.status);

      // Forward response headers
      flaskResponse.headers.forEach((value, key) => {
        // Skip certain headers that shouldn't be forwarded
        const lowerKey = key.toLowerCase();
        if (lowerKey !== 'transfer-encoding' && lowerKey !== 'connection') {
          res.setHeader(key, value);
        }
      });

      // Forward response body
      const contentType = flaskResponse.headers.get('content-type') || '';
      
      if (contentType.includes('application/json')) {
        // JSON response
        const data = await flaskResponse.json();
        res.json(data);
      } else if (contentType.includes('text/')) {
        // Text response
        const text = await flaskResponse.text();
        res.send(text);
      } else {
        // Binary response (e.g., images, files)
        const buffer = await flaskResponse.arrayBuffer();
        res.send(Buffer.from(buffer));
      }

    } catch (error) {
      // Error during proxying
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      console.error(`[Proxy] Error proxying request to Flask: ${errorMessage}`);

      const errorResponse: FlaskUnavailableError = {
        error: 'Proxy error',
        message: 'Failed to communicate with the deepfake detection service',
        suggestion: 'Check Flask server logs for errors',
        timestamp: new Date().toISOString(),
      };

      return res.status(503).json(errorResponse);
    }
  };
}
