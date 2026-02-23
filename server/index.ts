import "dotenv/config";
import express from "express";
import cors from "cors";
import path from "path";
import { handleDemo } from "./routes/demo";
import scannerRoutes from "./routes/scanner";
import { PythonBridge } from "./python-bridge";
import { createDeepfakeProxy } from "./routes/deepfake-proxy";
import { loadFlaskConfig } from "./config";
import scamRoutes from "./routes/scams";
import generalReportRoutes from "./routes/generalReports";
import fraudDetectionRoutes from "./routes/fraudDetection";

export function createServer(pythonBridge?: PythonBridge) {
  const app = express();

  // Logging middleware
  app.use((req, res, next) => {
    console.log(`[Express] ${req.method} ${req.url}`);
    next();
  });

  // Middleware
  app.use(cors());
  app.use(express.json());
  app.use(express.urlencoded({ extended: true }));

  // Example API routes
  app.get("/api/ping", (_req, res) => {
    const ping = process.env.PING_MESSAGE ?? "CyberGuard API is running";
    res.json({
      message: ping,
      version: "1.0.1",
      timestamp: new Date().toISOString(),
      environment: process.env.VERCEL ? 'vercel' : 'local'
    });
  });

  app.get("/api/demo", handleDemo);

  // Scanner routes
  app.use("/api/scanner", scannerRoutes);

  // Scam alert routes (location-based)
  app.use("/api/scams", scamRoutes);

  // General scam report routes (non-location-based)
  app.use("/api/reports", generalReportRoutes);

  // Fraud detection routes (proxy to Python service)
  app.use("/api/fraud", fraudDetectionRoutes);

  // Serve uploaded files
  app.use("/uploads", express.static("uploads"));

  // Deepfake proxy routes (if PythonBridge is provided and NOT on Vercel)
  if (pythonBridge && !process.env.VERCEL) {
    const deepfakeProxy = createDeepfakeProxy({
      bridge: pythonBridge,
      targetUrl: pythonBridge.getFlaskUrl(),
    });

    // Register proxy middleware for all /api/deepfake/* routes
    app.use("/api/deepfake", deepfakeProxy);
  }

  return app;
}

/**
 * Start the Express server with Python Flask integration
 * 
 * This function:
 * 1. Loads Flask configuration from environment variables
 * 2. Creates and starts the PythonBridge
 * 3. Creates the Express app with the bridge
 * 4. Starts the Express server
 * 5. Sets up shutdown handlers
 */
export async function startServer(): Promise<void> {
  const PORT = parseInt(process.env.PORT || '8080', 10);

  console.log('='.repeat(60));
  console.log('Starting CyberGuard Backend Server');
  console.log('='.repeat(60));

  // Load Flask configuration
  const flaskConfig = loadFlaskConfig();

  // Create PythonBridge instance
  const pythonBridge = new PythonBridge({
    pythonScript: path.join(__dirname, '../python/api_server.py'),
    port: flaskConfig.port,
    host: flaskConfig.host,
    healthCheckTimeout: flaskConfig.healthCheckTimeout,
    healthCheckInterval: flaskConfig.healthCheckInterval,
    healthCheckMaxRetries: flaskConfig.healthCheckMaxRetries,
    shutdownTimeout: flaskConfig.shutdownTimeout,
    isDevelopment: flaskConfig.isDevelopment,
  });

  // Start Flask server (Skip on Vercel)
  if (!process.env.VERCEL) {
    try {
      await pythonBridge.start();
    } catch (error) {
      console.warn('[Server] Failed to start PythonBridge, but continuing...', error);
    }
  } else {
    console.log('[Server] Vercel detected, skipping PythonBridge startup');
  }

  // Create Express app with PythonBridge
  const app = createServer(pythonBridge);

  // Start Express server
  const server = app.listen(PORT, () => {
    console.log('='.repeat(60));
    console.log(`✓ Express server listening on port ${PORT}`);
    console.log(`✓ API available at: http://localhost:${PORT}/api`);
    if (pythonBridge.isFlaskReady()) {
      console.log(`✓ Deepfake API available at: http://localhost:${PORT}/api/deepfake`);
    }
    console.log('='.repeat(60));
  });

  // Set up shutdown handlers
  const shutdown = async (signal: string) => {
    console.log(`\n[Server] Received ${signal}, shutting down gracefully...`);

    // Stop accepting new connections
    server.close(() => {
      console.log('[Server] Express server closed');
    });

    // Stop Flask server
    await pythonBridge.stop();

    // Exit process
    console.log('[Server] Shutdown complete');
    process.exit(0);
  };

  // Register shutdown handlers
  process.on('SIGINT', () => shutdown('SIGINT'));
  process.on('SIGTERM', () => shutdown('SIGTERM'));

  // Handle uncaught errors
  process.on('uncaughtException', (error) => {
    console.error('[Server] Uncaught exception:', error);
    shutdown('uncaughtException');
  });

  process.on('unhandledRejection', (reason, promise) => {
    console.error('[Server] Unhandled rejection at:', promise, 'reason:', reason);
    shutdown('unhandledRejection');
  });
}
