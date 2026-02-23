/**
 * MySQL Database Connection
 * 
 * Manages database connection pool for the scam alert system
 */

import mysql from 'mysql2/promise';

// Database configuration from environment variables
const dbConfig = {
  host: process.env.DB_HOST || 'localhost',
  port: parseInt(process.env.DB_PORT || '3306', 10),
  user: process.env.DB_USER || 'root',
  password: process.env.DB_PASSWORD || '',
  database: process.env.DB_NAME || 'cybershakti',
  waitForConnections: true,
  connectionLimit: 10,
  queueLimit: 0,
  enableKeepAlive: true,
  keepAliveInitialDelay: 0,
};

// Create connection pool
let pool: mysql.Pool | null = null;

/**
 * Get database connection pool
 * Creates pool on first call, returns existing pool on subsequent calls
 */
export function getPool(): mysql.Pool {
  if (!pool) {
    pool = mysql.createPool(dbConfig);
    console.log('[Database] MySQL connection pool created');
  }
  return pool;
}

/**
 * Test database connection
 * Returns true if connection is successful, false otherwise
 */
export async function testConnection(): Promise<boolean> {
  try {
    const connection = await getPool().getConnection();
    await connection.ping();
    connection.release();
    console.log('[Database] ✓ MySQL connection successful');
    return true;
  } catch (error) {
    console.error('[Database] ✗ MySQL connection failed:', error instanceof Error ? error.message : 'Unknown error');
    return false;
  }
}

/**
 * Close database connection pool
 * Should be called on application shutdown
 */
export async function closePool(): Promise<void> {
  if (pool) {
    await pool.end();
    pool = null;
    console.log('[Database] MySQL connection pool closed');
  }
}

/**
 * Execute a query with automatic connection management
 * 
 * @param query - SQL query string
 * @param params - Query parameters (for prepared statements)
 * @returns Query results
 */
export async function executeQuery<T = any>(
  query: string,
  params?: any[]
): Promise<T> {
  const connection = await getPool().getConnection();
  try {
    const [results] = await connection.execute(query, params);
    return results as T;
  } finally {
    connection.release();
  }
}

/**
 * Execute a stored procedure
 * 
 * @param procedureName - Name of the stored procedure
 * @param params - Procedure parameters
 * @returns Procedure results
 */
export async function callProcedure<T = any>(
  procedureName: string,
  params: any[]
): Promise<T> {
  const placeholders = params.map(() => '?').join(', ');
  const query = `CALL ${procedureName}(${placeholders})`;
  const results = await executeQuery(query, params);
  // Stored procedures return an array of result sets
  return Array.isArray(results) && results.length > 0 ? results[0] : results;
}
