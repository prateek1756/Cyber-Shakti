/**
 * Configuration module for Flask server integration
 * 
 * This module loads and validates environment variables for the Python Flask
 * backend integration, providing type-safe configuration with sensible defaults.
 * 
 * Validates: Requirements 10.1, 10.2, 10.3, 10.4, 10.5
 */

/**
 * Flask server configuration interface
 */
export interface FlaskConfig {
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
  /** Whether running in development mode */
  isDevelopment: boolean;
}

/**
 * Default configuration values
 */
const DEFAULT_CONFIG: FlaskConfig = {
  port: 5001,
  host: '0.0.0.0',
  healthCheckTimeout: 10000,
  healthCheckInterval: 500,
  healthCheckMaxRetries: 20,
  shutdownTimeout: 5000,
  isDevelopment: process.env.NODE_ENV !== 'production',
};

/**
 * Parse an integer from environment variable with validation
 * 
 * @param value - The environment variable value
 * @param defaultValue - The default value to use if parsing fails
 * @param name - The name of the environment variable (for logging)
 * @returns The parsed integer or default value
 */
function parseIntWithDefault(
  value: string | undefined,
  defaultValue: number,
  name: string
): number {
  if (value === undefined) {
    return defaultValue;
  }

  const parsed = parseInt(value, 10);
  
  if (isNaN(parsed) || parsed <= 0) {
    console.warn(
      `[Config] Invalid value for ${name}: "${value}". Using default: ${defaultValue}`
    );
    return defaultValue;
  }

  return parsed;
}

/**
 * Parse a string from environment variable with validation
 * 
 * @param value - The environment variable value
 * @param defaultValue - The default value to use if empty
 * @returns The parsed string or default value
 */
function parseStringWithDefault(
  value: string | undefined,
  defaultValue: string
): string {
  if (value === undefined || value.trim() === '') {
    return defaultValue;
  }

  return value.trim();
}

/**
 * Load and validate Flask configuration from environment variables
 * 
 * Supported environment variables:
 * - FLASK_PORT: Flask server port (default: 5001)
 * - FLASK_HOST: Flask server host (default: 0.0.0.0)
 * - FLASK_HEALTH_TIMEOUT: Health check timeout in ms (default: 10000)
 * - FLASK_HEALTH_INTERVAL: Health check interval in ms (default: 500)
 * - FLASK_HEALTH_MAX_RETRIES: Maximum health check retries (default: 20)
 * - FLASK_SHUTDOWN_TIMEOUT: Shutdown timeout in ms (default: 5000)
 * - NODE_ENV: Environment mode (development/production)
 * 
 * @returns Validated Flask configuration object
 */
export function loadFlaskConfig(): FlaskConfig {
  const config: FlaskConfig = {
    port: parseIntWithDefault(
      process.env.FLASK_PORT,
      DEFAULT_CONFIG.port,
      'FLASK_PORT'
    ),
    host: parseStringWithDefault(
      process.env.FLASK_HOST,
      DEFAULT_CONFIG.host
    ),
    healthCheckTimeout: parseIntWithDefault(
      process.env.FLASK_HEALTH_TIMEOUT,
      DEFAULT_CONFIG.healthCheckTimeout,
      'FLASK_HEALTH_TIMEOUT'
    ),
    healthCheckInterval: parseIntWithDefault(
      process.env.FLASK_HEALTH_INTERVAL,
      DEFAULT_CONFIG.healthCheckInterval,
      'FLASK_HEALTH_INTERVAL'
    ),
    healthCheckMaxRetries: parseIntWithDefault(
      process.env.FLASK_HEALTH_MAX_RETRIES,
      DEFAULT_CONFIG.healthCheckMaxRetries,
      'FLASK_HEALTH_MAX_RETRIES'
    ),
    shutdownTimeout: parseIntWithDefault(
      process.env.FLASK_SHUTDOWN_TIMEOUT,
      DEFAULT_CONFIG.shutdownTimeout,
      'FLASK_SHUTDOWN_TIMEOUT'
    ),
    isDevelopment: process.env.NODE_ENV !== 'production',
  };

  // Log configuration in development mode
  if (config.isDevelopment) {
    console.log('[Config] Flask configuration loaded:', {
      port: config.port,
      host: config.host,
      healthCheckTimeout: `${config.healthCheckTimeout}ms`,
      healthCheckInterval: `${config.healthCheckInterval}ms`,
      healthCheckMaxRetries: config.healthCheckMaxRetries,
      shutdownTimeout: `${config.shutdownTimeout}ms`,
      isDevelopment: config.isDevelopment,
    });
  }

  return config;
}

/**
 * Export default configuration for testing and reference
 */
export { DEFAULT_CONFIG };
