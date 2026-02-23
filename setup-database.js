/**
 * Database Setup Script
 * 
 * Creates the cybershakti database and all required tables
 */

import 'dotenv/config';
import mysql from 'mysql2/promise';
import { readFileSync } from 'fs';

async function setupDatabase() {
  console.log('='.repeat(60));
  console.log('CyberShakti - Database Setup');
  console.log('='.repeat(60));
  
  const config = {
    host: process.env.DB_HOST || 'localhost',
    port: parseInt(process.env.DB_PORT || '3306', 10),
    user: process.env.DB_USER || 'root',
    password: process.env.DB_PASSWORD || '',
    multipleStatements: true,
  };
  
  console.log('\nConnecting to MySQL...');
  console.log(`  Host: ${config.host}`);
  console.log(`  Port: ${config.port}`);
  console.log(`  User: ${config.user}`);
  
  let connection;
  
  try {
    // Connect to MySQL (without specifying database)
    connection = await mysql.createConnection(config);
    console.log('✓ Connected to MySQL successfully!\n');
    
    // Read schema file
    console.log('Reading database schema...');
    const schema = readFileSync('database/schema-simple.sql', 'utf8');
    console.log('✓ Schema file loaded\n');
    
    // Execute schema
    console.log('Creating database and tables...');
    await connection.query(schema);
    console.log('✓ Database and tables created successfully!\n');
    
    console.log('='.repeat(60));
    console.log('✓ DATABASE SETUP COMPLETE!');
    console.log('='.repeat(60));
    console.log('\nDatabase: cybershakti');
    console.log('Tables created:');
    console.log('  - scam_reports (location-based)');
    console.log('  - general_scam_reports (general reporting)');
    console.log('  - scam_evidence_files (file uploads)');
    console.log('  - admin_users (admin management)');
    console.log('\nSample data has been inserted for testing.');
    console.log('='.repeat(60));
    
  } catch (error) {
    console.error('\n✗ ERROR:', error.message);
    console.log('\nTROUBLESHOOTING:');
    console.log('1. Make sure MySQL is running');
    console.log('2. Verify credentials in .env file');
    console.log('3. Check if you have permission to create databases');
    process.exit(1);
  } finally {
    if (connection) {
      await connection.end();
    }
  }
}

setupDatabase();
