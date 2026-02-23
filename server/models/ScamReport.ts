/**
 * ScamReport Model
 * 
 * Handles database operations for scam reports
 */

import { executeQuery, callProcedure } from '../db/connection';
import { RowDataPacket, ResultSetHeader } from 'mysql2';

export interface ScamReport {
  id?: number;
  title: string;
  description: string;
  scam_type: 'phishing' | 'phone_scam' | 'fake_website' | 'identity_theft' | 
              'investment_fraud' | 'romance_scam' | 'tech_support' | 'other';
  latitude: number;
  longitude: number;
  reporter_ip?: string;
  status?: 'pending' | 'verified' | 'rejected';
  created_at?: Date;
  updated_at?: Date;
}

export interface NearbyScamReport extends ScamReport {
  distance_km: number;
}

export class ScamReportModel {
  /**
   * Get nearby scam reports within specified radius
   * 
   * @param latitude - User's latitude
   * @param longitude - User's longitude
   * @param radiusKm - Search radius in kilometers (default: 10)
   * @returns Array of nearby scam reports with distance
   */
  static async getNearby(
    latitude: number,
    longitude: number,
    radiusKm: number = 10
  ): Promise<NearbyScamReport[]> {
    try {
      // Use stored procedure for optimized geospatial query
      const results = await callProcedure<RowDataPacket[]>(
        'GetNearbyScams',
        [latitude, longitude, radiusKm]
      );
      
      return results as NearbyScamReport[];
    } catch (error) {
      console.error('[ScamReportModel] Error fetching nearby scams:', error);
      throw new Error('Failed to fetch nearby scam reports');
    }
  }

  /**
   * Create a new scam report
   * 
   * @param report - Scam report data
   * @returns Created report ID
   */
  static async create(report: ScamReport): Promise<number> {
    const query = `
      INSERT INTO scam_reports 
      (title, description, scam_type, latitude, longitude, reporter_ip, status)
      VALUES (?, ?, ?, ?, ?, ?, ?)
    `;
    
    const params = [
      report.title,
      report.description,
      report.scam_type,
      report.latitude,
      report.longitude,
      report.reporter_ip || null,
      report.status || 'pending'
    ];

    try {
      const result = await executeQuery<ResultSetHeader>(query, params);
      return result.insertId;
    } catch (error) {
      console.error('[ScamReportModel] Error creating scam report:', error);
      throw new Error('Failed to create scam report');
    }
  }

  /**
   * Get scam report by ID
   * 
   * @param id - Report ID
   * @returns Scam report or null if not found
   */
  static async getById(id: number): Promise<ScamReport | null> {
    const query = `
      SELECT id, title, description, scam_type, latitude, longitude, 
             reporter_ip, status, created_at, updated_at
      FROM scam_reports
      WHERE id = ?
    `;

    try {
      const results = await executeQuery<RowDataPacket[]>(query, [id]);
      return results.length > 0 ? (results[0] as ScamReport) : null;
    } catch (error) {
      console.error('[ScamReportModel] Error fetching scam report:', error);
      throw new Error('Failed to fetch scam report');
    }
  }

  /**
   * Get all scam reports (with pagination)
   * 
   * @param limit - Number of reports to return
   * @param offset - Offset for pagination
   * @param status - Filter by status (optional)
   * @returns Array of scam reports
   */
  static async getAll(
    limit: number = 50,
    offset: number = 0,
    status?: 'pending' | 'verified' | 'rejected'
  ): Promise<ScamReport[]> {
    let query = `
      SELECT id, title, description, scam_type, latitude, longitude, 
             reporter_ip, status, created_at, updated_at
      FROM scam_reports
    `;

    const params: any[] = [];

    if (status) {
      query += ' WHERE status = ?';
      params.push(status);
    }

    query += ' ORDER BY created_at DESC LIMIT ? OFFSET ?';
    params.push(limit, offset);

    try {
      const results = await executeQuery<RowDataPacket[]>(query, params);
      return results as ScamReport[];
    } catch (error) {
      console.error('[ScamReportModel] Error fetching scam reports:', error);
      throw new Error('Failed to fetch scam reports');
    }
  }

  /**
   * Update scam report status (for moderation)
   * 
   * @param id - Report ID
   * @param status - New status
   * @returns True if updated successfully
   */
  static async updateStatus(
    id: number,
    status: 'pending' | 'verified' | 'rejected'
  ): Promise<boolean> {
    const query = `
      UPDATE scam_reports
      SET status = ?, updated_at = CURRENT_TIMESTAMP
      WHERE id = ?
    `;

    try {
      const result = await executeQuery<ResultSetHeader>(query, [status, id]);
      return result.affectedRows > 0;
    } catch (error) {
      console.error('[ScamReportModel] Error updating scam report status:', error);
      throw new Error('Failed to update scam report status');
    }
  }

  /**
   * Delete scam report
   * 
   * @param id - Report ID
   * @returns True if deleted successfully
   */
  static async delete(id: number): Promise<boolean> {
    const query = 'DELETE FROM scam_reports WHERE id = ?';

    try {
      const result = await executeQuery<ResultSetHeader>(query, [id]);
      return result.affectedRows > 0;
    } catch (error) {
      console.error('[ScamReportModel] Error deleting scam report:', error);
      throw new Error('Failed to delete scam report');
    }
  }

  /**
   * Get scam statistics
   * 
   * @returns Statistics object
   */
  static async getStatistics(): Promise<{
    total: number;
    pending: number;
    verified: number;
    rejected: number;
    byType: Record<string, number>;
  }> {
    const query = `
      SELECT 
        COUNT(*) as total,
        SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END) as pending,
        SUM(CASE WHEN status = 'verified' THEN 1 ELSE 0 END) as verified,
        SUM(CASE WHEN status = 'rejected' THEN 1 ELSE 0 END) as rejected,
        scam_type,
        COUNT(*) as type_count
      FROM scam_reports
      GROUP BY scam_type WITH ROLLUP
    `;

    try {
      const results = await executeQuery<RowDataPacket[]>(query);
      
      // Parse results
      const stats = {
        total: 0,
        pending: 0,
        verified: 0,
        rejected: 0,
        byType: {} as Record<string, number>
      };

      results.forEach((row: any) => {
        if (row.scam_type === null) {
          // Rollup row with totals
          stats.total = row.total;
          stats.pending = row.pending;
          stats.verified = row.verified;
          stats.rejected = row.rejected;
        } else {
          // Individual type counts
          stats.byType[row.scam_type] = row.type_count;
        }
      });

      return stats;
    } catch (error) {
      console.error('[ScamReportModel] Error fetching statistics:', error);
      throw new Error('Failed to fetch statistics');
    }
  }
}
