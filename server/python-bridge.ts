/**
 * PythonBridge Module
 * 
 * This module manages the Python Flask server as a child process of the Node.js
 * Express server. It handles process lifecycle management, health monitoring,
 * log forwarding, and graceful shutdown.
 * 
 * Validates: Requirements 1.2, 2.5
 */

import { ChildProcess, execSync, spawn } from 'child_process';
import { existsSync } from 'fs';
import { resolve } from 'path';

/**
 * Configuration interface for PythonBridge
 */
export interface PythonBridgeConfig {
  /** Path to the Python Flask server script (api_server.py) */
  pythonScript: string;
  /** Flask server port (default: 5001) */
  port: number;
  /** Flask server host (default: 0.0.0.0) */
  host: string;
  /** Maximum time to wait for Flask health check in milliseconds (default: 10000) */
  healthCheckTimeout: number;
  /** Interval between health check retries in milliseconds (default: 500) */
  healthCheckInterval: number;
  /** Maximum number of health check retry attempts (default: 20) */
  healthCheckMaxRetries: number;
  /** Maximum time to wait for graceful shutdown in milliseconds (default: 5000) */
  shutdownTimeout: number;
  /** Whether running in development mode (enables verbose logging) */
  isDevelopment: boolean;
  /** Path for health check (default: /api/deepfake/stats) */
  healthCheckPath: string;
}

/**
 * Internal state interface for PythonBridge
 */
export interface BridgeState {
  /** The Flask child process instance */
  process: ChildProcess | null;
  /** Whether the Flask server is ready to accept requests */
  isReady: boolean;
  /** Timestamp when the Flask process was started */
  startTime: number | null;
  /** Timestamp of the last successful health check */
  lastHealthCheck: number | null;
}

/**
 * PythonBridge class manages the Flask server lifecycle
 * 
 * This class is responsible for:
 * - Spawning and managing the Flask server child process
 * - Monitoring Flask server health
 * - Capturing and forwarding Flask logs
 * - Handling graceful shutdown
 * - Detecting Python environment issues
 */
export class PythonBridge {
  /** Bridge configuration */
  private config: PythonBridgeConfig;

  /** Current bridge state */
  private state: BridgeState;

  /**
   * Create a new PythonBridge instance
   * 
   * @param config - Partial configuration object (merged with defaults)
   */
  constructor(config: Partial<PythonBridgeConfig>) {
    // Merge provided config with defaults
    this.config = {
      pythonScript: config.pythonScript || '',
      port: config.port || 5001,
      host: config.host || '0.0.0.0',
      healthCheckTimeout: config.healthCheckTimeout || 10000,
      healthCheckInterval: config.healthCheckInterval || 500,
      healthCheckMaxRetries: config.healthCheckMaxRetries || 20,
      shutdownTimeout: config.shutdownTimeout || 5000,
      isDevelopment: config.isDevelopment !== undefined ? config.isDevelopment : process.env.NODE_ENV !== 'production',
      healthCheckPath: config.healthCheckPath || '/api/deepfake/stats',
    };

    // Initialize state
    this.state = {
      process: null,
      isReady: false,
      startTime: null,
      lastHealthCheck: null,
    };
  }

  /**
   * Start the Flask server
   * 
   * This method:
   * 1. Validates the Python environment
   * 2. Spawns the Flask child process
   * 3. Sets up process handlers for logs and exit
   * 4. Waits for the Flask server to become healthy
   * 5. Logs confirmation or error messages
   * 
   * @throws Error if Flask fails to start (in strict mode)
   */
  async start(): Promise<void> {
    console.log('[Python] Starting Flask server integration...');

    // Step 1: Validate environment
    const validation = this.validateEnvironment();

    if (!validation.valid) {
      // Environment validation failed
      console.error(`[Python] ❌ ${validation.error}`);
      console.error('[Python] Express server will continue running without Flask integration');
      console.error('[Python] Deepfake endpoints will return 503 Service Unavailable');
      return; // Don't throw - gracefully degrade
    }

    console.log('[Python] ✓ Environment validation passed');

    try {
      // Step 2: Spawn Flask process
      const proc = this.spawnFlaskProcess();

      // Step 3: Set up process handlers
      this.setupProcessHandlers(proc);

      // Step 4: Wait for health check
      await this.waitForHealth();

      // Step 5: Log final status
      if (this.state.isReady) {
        console.log('[Python] ✓ Flask server started successfully');
        console.log(`[Python] Flask URL: ${this.getFlaskUrl()}`);
      } else {
        console.error('[Python] ❌ Flask server failed to become ready');
        console.error('[Python] Express server will continue running without Flask integration');
      }

    } catch (error) {
      // Unexpected error during startup
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      console.error(`[Python] ❌ Failed to start Flask server: ${errorMessage}`);
      console.error('[Python] Express server will continue running without Flask integration');

      // Clean up state
      this.state.isReady = false;
      this.state.process = null;
    }
  }

  /**
   * Check if the Flask server is ready to accept requests
   * 
   * @returns true if Flask is ready, false otherwise
   */
  isFlaskReady(): boolean {
    return this.state.isReady && this.state.process !== null && this.state.process.exitCode === null;
  }

  /**
   * Get the Flask server URL
   * 
   * @returns Flask server URL (e.g., "http://localhost:5001")
   */
  getFlaskUrl(): string {
    return `http://127.0.0.1:${this.config.port}`;
  }

  /**
   * Stop the Flask server gracefully
   * 
   * This method:
   * 1. Sends SIGTERM to the Flask process
   * 2. Waits up to shutdownTimeout for the process to exit
   * 3. Forcefully kills the process if it doesn't exit in time
   * 4. Cleans up process references
   * 5. Logs shutdown progress
   */
  async stop(): Promise<void> {
    if (!this.state.process) {
      console.log('[Python] No Flask process to stop');
      return;
    }

    console.log('[Python] Stopping Flask server...');

    const proc = this.state.process;
    const pid = proc.pid;

    // Mark as not ready immediately
    this.state.isReady = false;

    try {
      // Send SIGTERM for graceful shutdown
      const isWindows = process.platform === 'win32';

      if (isWindows) {
        // On Windows, use taskkill
        console.log(`[Python] Sending termination signal to Flask process (PID: ${pid})...`);
        proc.kill('SIGTERM'); // Node.js handles this on Windows
      } else {
        // On Unix, send SIGTERM
        console.log(`[Python] Sending SIGTERM to Flask process (PID: ${pid})...`);
        proc.kill('SIGTERM');
      }

      // Wait for process to exit with timeout
      const exitPromise = new Promise<void>((resolve) => {
        proc.once('exit', () => {
          resolve();
        });
      });

      const timeoutPromise = new Promise<void>((resolve) => {
        setTimeout(() => {
          resolve();
        }, this.config.shutdownTimeout);
      });

      // Race between exit and timeout
      await Promise.race([exitPromise, timeoutPromise]);

      // Check if process is still running
      if (proc.exitCode === null && !proc.killed) {
        // Process didn't exit in time, force kill
        console.warn(`[Python] Flask process did not exit within ${this.config.shutdownTimeout}ms, forcing kill...`);
        proc.kill('SIGKILL');

        // Wait a bit more for forced kill
        await new Promise(resolve => setTimeout(resolve, 500));
      }

      console.log('[Python] ✓ Flask server stopped');

    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      console.error(`[Python] Error stopping Flask server: ${errorMessage}`);
    } finally {
      // Clean up state
      this.state.process = null;
      this.state.isReady = false;
    }
  }

  /**
   * Spawn the Flask child process
   * 
   * This method:
   * 1. Detects the appropriate Python command
   * 2. Spawns the Flask process with appropriate arguments
   * 3. Configures stdio pipes for log capture
   * 4. Stores the process reference
   * 
   * @returns The spawned child process
   * @private
   */
  private spawnFlaskProcess(): ChildProcess {
    // Detect the appropriate Python command for the platform
    const pythonCommand = this.detectPythonCommand();

    // Resolve the absolute path to the Flask script
    const scriptPath = resolve(this.config.pythonScript);

    // Prepare command-line arguments for Flask
    // Pass port and host as command-line arguments
    const args = [
      scriptPath,
      '--port', this.config.port.toString(),
      '--host', this.config.host,
    ];

    // Add debug flag in development mode
    if (this.config.isDevelopment) {
      args.push('--debug');
    }

    // Prepare environment variables
    // Pass configuration through environment as well (Flask can use either)
    const env = {
      ...process.env,
      FLASK_PORT: this.config.port.toString(),
      FLASK_HOST: this.config.host,
      FLASK_ENV: this.config.isDevelopment ? 'development' : 'production',
    };

    // Determine working directory (parent of the script)
    const cwd = resolve(scriptPath, '..');

    // Spawn the Flask process
    const proc = spawn(pythonCommand, args, {
      cwd,
      env,
      stdio: ['ignore', 'pipe', 'pipe'], // stdin ignored, stdout and stderr piped
    });

    // Store process reference in state
    this.state.process = proc;
    this.state.startTime = Date.now();

    // Log spawn information
    console.log(`[Python] Spawning Flask server: ${pythonCommand} ${args.join(' ')}`);
    console.log(`[Python] Working directory: ${cwd}`);
    console.log(`[Python] Process PID: ${proc.pid}`);

    return proc;
  }


  /**
   * Detect the appropriate Python command for the current platform
   * 
   * This method checks for:
   * - `python` command (Windows, some Unix systems)
   * - `python3` command (Unix systems)
   * 
   * @returns The Python command to use
   * @throws Error if Python is not found
   * @private
   */
  private detectPythonCommand(): string {
    const isWindows = process.platform === 'win32';

    // On Windows, use 'where' command; on Unix, use 'which'
    const checkCommand = isWindows ? 'where' : 'which';

    // Commands to check in order of preference
    const commandsToCheck = isWindows ? ['python'] : ['python3', 'python'];

    for (const cmd of commandsToCheck) {
      try {
        // Try to find the command in PATH
        execSync(`${checkCommand} ${cmd}`, {
          stdio: 'pipe',
          encoding: 'utf-8'
        });

        // If execSync doesn't throw, the command exists
        return cmd;
      } catch (error) {
        // Command not found, try next one
        continue;
      }
    }

    // No Python command found
    throw new Error(
      'Python executable not found in system PATH. ' +
      'Please install Python 3.8+ and ensure it is added to your PATH. ' +
      'Visit https://www.python.org/downloads/ for installation instructions.'
    );
  }

  /**
   * Validate the Python environment
   * 
   * This method checks:
   * 1. Python is available in the system PATH
   * 2. The Flask server script exists
   * 
   * @returns Validation result with error messages if any
   * @private
   */
  private validateEnvironment(): { valid: boolean; error?: string } {
    // Check 1: Verify Python is available
    try {
      this.detectPythonCommand();
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      return {
        valid: false,
        error: `Python environment validation failed: ${errorMessage}`,
      };
    }

    // Check 2: Verify Flask server script exists
    const scriptPath = resolve(this.config.pythonScript);
    if (!existsSync(scriptPath)) {
      return {
        valid: false,
        error: `Flask server script not found at: ${scriptPath}. ` +
          'Please ensure the Python backend is properly set up. ' +
          'Expected file: python/api_server.py',
      };
    }

    // All checks passed
    return { valid: true };
  }

  /**
   * Set up process handlers for the Flask child process
   * 
   * This method attaches handlers for:
   * - stdout (Flask logs)
   * - stderr (Flask errors)
   * - exit event
   * 
   * @param proc - The Flask child process
   * @private
   */
  private setupProcessHandlers(proc: ChildProcess): void {
    // Handle stdout - Flask application logs
    if (proc.stdout) {
      proc.stdout.on('data', (data: Buffer) => {
        const output = data.toString().trim();
        if (output) {
          // In development mode, log all Flask output
          // In production mode, only log if it looks important
          if (this.config.isDevelopment) {
            console.log(`[Python] ${output}`);
          } else {
            // In production, only log warnings and errors
            const lowerOutput = output.toLowerCase();
            if (lowerOutput.includes('error') ||
              lowerOutput.includes('warning') ||
              lowerOutput.includes('exception')) {
              console.log(`[Python] ${output}`);
            }
          }
        }
      });
    }

    // Handle stderr - Flask errors and warnings
    if (proc.stderr) {
      proc.stderr.on('data', (data: Buffer) => {
        const output = data.toString().trim();
        if (output) {
          // Always log stderr output with [Python] prefix
          console.error(`[Python] ${output}`);
        }
      });
    }

    // Handle process exit
    proc.on('exit', (code, signal) => {
      this.handleProcessExit(code, signal);
    });

    // Handle process errors (e.g., failed to spawn)
    proc.on('error', (error: Error) => {
      console.error(`[Python] Process error: ${error.message}`);
      console.error('[Python] Failed to spawn Flask server process');

      // Update state
      this.state.isReady = false;
      this.state.process = null;
    });
  }

  /**
   * Wait for the Flask server to become healthy
   * 
   * This method:
   * 1. Performs health checks at configured intervals
   * 2. Retries up to the configured maximum attempts
   * 3. Updates the isReady state on success
   * 4. Logs warnings with troubleshooting steps on failure
   * 
   * @private
   */
  private async waitForHealth(): Promise<void> {
    console.log(`[Python] Waiting for Flask server to become ready...`);
    console.log(`[Python] Health check URL: http://localhost:${this.config.port}${this.config.healthCheckPath}`);
    console.log(`[Python] Max retries: ${this.config.healthCheckMaxRetries}, Interval: ${this.config.healthCheckInterval}ms`);

    let attempts = 0;
    const maxAttempts = this.config.healthCheckMaxRetries;
    const interval = this.config.healthCheckInterval;

    while (attempts < maxAttempts) {
      attempts++;

      // Perform health check
      const isHealthy = await this.performHealthCheck();

      if (isHealthy) {
        // Flask is ready!
        this.state.isReady = true;
        const elapsedTime = this.state.startTime ? Date.now() - this.state.startTime : 0;
        console.log(`[Python] Flask server is ready! (took ${(elapsedTime / 1000).toFixed(2)}s, ${attempts} attempts)`);
        return;
      }

      // Check if process has exited
      if (!this.state.process || this.state.process.exitCode !== null) {
        console.error('[Python] Flask process exited during health check');
        console.error('[Python] Health check failed: Flask server is not running');
        return;
      }

      // Log progress periodically
      if (attempts % 5 === 0 || attempts === 1) {
        console.log(`[Python] Health check attempt ${attempts}/${maxAttempts}...`);
      }

      // Wait before next attempt
      if (attempts < maxAttempts) {
        await new Promise(resolve => setTimeout(resolve, interval));
      }
    }

    // Health check failed after all retries
    console.warn('[Python] ⚠️  Flask server health check failed after maximum retries');
    console.warn('[Python] The Flask server may not be ready or may have failed to start');
    console.warn('[Python] Troubleshooting steps:');
    console.warn('[Python]   1. Check the Flask server logs above for errors');
    console.warn('[Python]   2. Ensure Python dependencies are installed: cd python && pip install -r requirements.txt');
    console.warn('[Python]   3. Verify port ' + this.config.port + ' is not in use by another process');
    console.warn('[Python]   4. Try manually starting Flask: cd python && python api_server.py');
    console.warn('[Python] Express server will continue running, but deepfake endpoints will return 503 errors');
  }

  /**
   * Perform a single health check
   * 
   * This method makes an HTTP GET request to the Flask stats endpoint
   * and returns whether the server is responsive.
   * 
   * @returns true if health check succeeds, false otherwise
   * @private
   */
  private async performHealthCheck(): Promise<boolean> {
    try {
      const url = `http://localhost:${this.config.port}${this.config.healthCheckPath}`;

      // Make HTTP GET request with timeout
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 2000); // 2 second timeout per request

      const response = await fetch(url, {
        method: 'GET',
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      // Check if response is successful
      if (response.ok) {
        // Try to parse JSON to ensure Flask is responding correctly
        const data = await response.json();

        // Verify response has expected structure
        if (typeof data === 'object' && data !== null) {
          this.state.lastHealthCheck = Date.now();
          return true;
        }
      }

      return false;
    } catch (error) {
      // Health check failed (network error, timeout, etc.)
      // This is expected during startup, so we don't log errors here
      return false;
    }
  }

  /**
   * Handle Flask process exit
   * 
   * This method:
   * 1. Logs exit code and signal information
   * 2. Updates bridge state
   * 3. Provides troubleshooting guidance
   * 
   * @param code - Exit code (null if killed by signal)
   * @param signal - Signal that caused exit (null if exited normally)
   * @private
   */
  private handleProcessExit(code: number | null, signal: string | null): void {
    // Update state immediately
    this.state.isReady = false;
    this.state.process = null;

    // Determine exit reason and log appropriately
    if (signal) {
      // Process was killed by a signal
      console.log(`[Python] Flask server process terminated by signal: ${signal}`);
    } else if (code === 0) {
      // Normal exit
      console.log('[Python] Flask server process exited normally');
    } else if (code !== null) {
      // Abnormal exit with error code
      console.error(`[Python] Flask server process exited with code: ${code}`);

      // Provide troubleshooting guidance based on exit code
      if (code === 1) {
        console.error('[Python] Troubleshooting: This usually indicates a Python error.');
        console.error('[Python] Check the error messages above for details.');
        console.error('[Python] Common causes:');
        console.error('[Python]   - Missing Python dependencies (run: cd python && pip install -r requirements.txt)');
        console.error('[Python]   - Syntax errors in Python code');
        console.error('[Python]   - Port already in use (check if another process is using port ' + this.config.port + ')');
      } else if (code === 2) {
        console.error('[Python] Troubleshooting: This may indicate a command-line argument error.');
        console.error('[Python] Check that the Flask server script accepts the provided arguments.');
      } else {
        console.error('[Python] Troubleshooting: Unexpected exit code.');
        console.error('[Python] Check the Flask server logs above for error details.');
        console.error('[Python] You may need to restart the development server: pnpm dev');
      }
    } else {
      // Unknown exit reason
      console.error('[Python] Flask server process exited unexpectedly');
    }

    // Log runtime duration if available
    if (this.state.startTime) {
      const runtime = Date.now() - this.state.startTime;
      const runtimeSeconds = (runtime / 1000).toFixed(2);
      console.log(`[Python] Flask server ran for ${runtimeSeconds} seconds`);
    }
  }
}
