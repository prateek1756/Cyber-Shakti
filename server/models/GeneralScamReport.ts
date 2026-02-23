/**
 * GeneralScamReport Model
 * 
 * Handles database operations for general (non-location-based) scam reports
 */

import { executeQuery } from '../db/connection';
import { RowDataPacket, ResultSetHeader } from 'mysql2';

export interface GeneralScamReport {
  id?: number;
  title: string;
  description: string;
  scam_type: 'phishing' | 'phone_scam' | 'fake_website' | 'identity_theft' | 
              'investment_fraud' | 'romance_scam' | 'tech_support' | 
              'online_shopping' | 'job_scam' | 'lottery_scam' | 'charity_scam' | 'other';
  severity_level: 'low' | 'medium' | 'high' | 'critical';
  reporter_name?: string;
  reporter_email?: string;
  reporter_phone?: string;
  evidence_url?: string;
  evidence_filename?: string;
  reporter_ip?: string;
  status?: 'pending' | 'approved' | 'rejected' | 'under_review';
  admin_notes?: string;
  reviewed_by?: number;
  reviewed_at?: Date;
  created_at?: Date;
  updated_at?: Date;
}

export interface EvidenceFile {
  id?: number;
  report_id: number;
  filename: string;
  original_filename: string;
  file_path: string;
  file_size: number;
  mime_type: string;
  uploaded_at?: Date;
}

export class GeneralScamReportModel {
  /**
   * Create a new general scam report
   * 
   * @param report - Scam report data
   * @returns Created report ID
   */
  static async create(report: GeneralScamReport): Promise<number> {
    const query = `
      INSERT INTO general_scam_reports 
      (title, description, scam_type, severity_level, reporter_name, 
       reporter_email, reporter_phone, evidence_url, evidence_filename, 
       reporter_ip, status)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `;
    
    const params = [
      report.title,
      report.description,
      report.scam_type,
      report.severity_level,
      report.reporter_name || null,
      report.reporter_email || null,
      report.reporter_phone || null,
      report.evidence_url || null,
      report.evidence_filename || null,
      report.reporter_ip || null,
      report.status || 'pending'
    ];

    try {
      const result = await executeQuery<ResultSetHeader>(query, params);
      return result.insertId;
    } catch (error) {
      console.error('[GeneralScamReportModel] Error creating report:', error);
      throw new Error('Failed to create scam report');
    }
  }

  /**
   * Get report by ID
   * 
   * @param id - Report ID
   * @returns Scam report or null if not found
   */
  static async getById(id: number): Promise<GeneralScamReport | null> {
    const query = `
      SELECT id, title, description, scam_type, severity_level,
             reporter_name, reporter_email, reporter_phone,
             evidence_url, evidence_filename, reporter_ip, status,
             admin_notes, reviewed_by, reviewed_at, created_at, updated_at
      FROM general_scam_reports
      WHERE id = ?
    `;

    try {
      const results = await executeQuery<RowDataPacket[]>(query, [id]);
      return results.length > 0 ? (results[0] as GeneralScamReport) : null;
    } catch (error) {
      console.error('[GeneralScamReportModel] Error fetching report:', error);
      throw new Error('Failed to fetch scam report');
    }
  }

  /**
   * Get all reports with filtering and pagination
   * 
   * @param limit - Number of reports to return
   * @param offset - Offset for pagination
   * @param filters - Optional filters (status, scam_type, severity_level)
   * @returns Array of scam reports
   */
  static async getAll(
    limit: number = 50,
    offset: number = 0,
    filters?: {
      status?: 'pending' | 'approved' | 'rejected' | 'under_review';
      scam_type?: string;
      severity_level?: 'low' | 'medium' | 'high' | 'critical';
    }
  ): Promise<GeneralScamReport[]> {
    let query = `
      SELECT id, title, description, scam_type, severity_level,
             reporter_name, reporter_email, reporter_phone,
             evidence_url, evidence_filename, status,
             admin_notes, reviewed_by, reviewed_at, created_at, updated_at
      FROM general_scam_reports
      WHERE 1=1
    `;

    const params: any[] = [];

    if (filters?.status) {
      query += ' AND status = ?';
      params.push(filters.status);
    }

    if (filters?.scam_type) {
      query += ' AND scam_type = ?';
      params.push(filters.scam_type);
    }

    if (filters?.severity_level) {
      query += ' AND severity_level = ?';
      params.push(filters.severity_level);
    }

    query += ' ORDER BY created_at DESC LIMIT ? OFFSET ?';
    params.push(limit, offset);

    try {
      const results = await executeQuery<RowDataPacket[]>(query, params);
      return results as GeneralScamReport[];
    } catch (error) {
      console.error('[GeneralScamReportModel] Error fetching reports:', error);
      throw new Error('Failed to fetch scam reports');
    }
  }

  /**
   * Update report status with admin notes
   * 
   * @param id - Report ID
   * @param status - New status
   * @param adminNotes - Optional admin notes
   * @param reviewedBy - Admin user ID
   * @returns True if updated successfully
   */
  static async updateStatus(
    id: number,
    status: 'pending' | 'approved' | 'rejected' | 'under_review',
    adminNotes?: string,
    reviewedBy?: number
  ): Promise<boolean> {
    const query = `
      UPDATE general_scam_reports
      SET status = ?, 
          admin_notes = ?,
          reviewed_by = ?,
          reviewed_at = CURRENT_TIMESTAMP,
          updated_at = CURRENT_TIMESTAMP
      WHERE id = ?
    `;

    try {
      const result = await executeQuery<ResultSetHeader>(
        query, 
        [status, adminNotes || null, reviewedBy || null, id]
      );
      return result.affectedRows > 0;
    } catch (error) {
      console.error('[GeneralScamReportModel] Error updating status:', error);
      throw new Error('Failed to update report status');
    }
  }

  /**
   * Delete report
   * 
   * @param id - Report ID
   * @returns True if deleted successfully
   */
  static async delete(id: number): Promise<boolean> {
    const query = 'DELETE FROM general_scam_reports WHERE id = ?';

    try {
      const result = await executeQuery<ResultSetHeader>(query, [id]);
      return result.affectedRows > 0;
    } catch (error) {
      console.error('[GeneralScamReportModel] Error deleting report:', error);
      throw new Error('Failed to delete scam report');
    }
  }

  /**
   * Get statistics
   * 
   * @returns Statistics object
   */
  static async getStatistics(): Promise<{
    total: number;
    pending: number;
    approved: number;
    rejected: number;
    under_review: number;
    byType: Record<string, number>;
    bySeverity: Record<string, number>;
  }> {
    const query = `
      SELECT 
        COUNT(*) as total,
        SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END) as pending,
        SUM(CASE WHEN status = 'approved' THEN 1 ELSE 0 END) as approved,
        SUM(CASE WHEN status = 'rejected' THEN 1 ELSE 0 END) as rejected,
        SUM(CASE WHEN status = 'under_review' THEN 1 ELSE 0 END) as under_review
      FROM general_scam_reports
    `;

    const typeQuery = `
      SELECT scam_type, COUNT(*) as count
      FROM general_scam_reports
      GROUP BY scam_type
    `;

    const severityQuery = `
      SELECT severity_level, COUNT(*) as count
      FROM general_scam_reports
      GROUP BY severity_level
    `;

    try {
      const [statsResults, typeResults, severityResults] = await Promise.all([
        executeQuery<RowDataPacket[]>(query),
        executeQuery<RowDataPacket[]>(typeQuery),
        executeQuery<RowDataPacket[]>(severityQuery)
      ]);

      const stats = statsResults[0] as any;
      const byType: Record<string, number> = {};
      const bySeverity: Record<string, number> = {};

      typeResults.forEach((row: any) => {
        byType[row.scam_type] = row.count;
      });

      severityResults.forEach((row: any) => {
        bySeverity[row.severity_level] = row.count;
      });

      return {
        total: stats.total || 0,
        pending: stats.pending || 0,
        approved: stats.approved || 0,
        rejected: stats.rejected || 0,
        under_review: stats.under_review || 0,
        byType,
        bySeverity
      };
    } catch (error) {
      console.error('[GeneralScamReportModel] Error fetching statistics:', error);
      throw new Error('Failed to fetch statistics');
    }
  }

  /**
   * Add evidence file record
   * 
   * @param evidence - Evidence file data
   * @returns Created evidence ID
   */
  static async addEvidence(evidence: EvidenceFile): Promise<number> {
    const query = `
      INSERT INTO scam_evidence_files
      (report_id, filename, original_filename, file_path, file_size, mime_type)
      VALUES (?, ?, ?, ?, ?, ?)
    `;

    const params = [
      evidence.report_id,
      evidence.filename,
      evidence.original_filename,
      evidence.file_path,
      evidence.file_size,
      evidence.mime_type
    ];

    try {
      const result = await executeQuery<ResultSetHeader>(query, params);
      return result.insertId;
    } catch (error) {
      console.error('[GeneralScamReportModel] Error adding evidence:', error);
      throw new Error('Failed to add evidence file');
    }
  }

  /**
   * Get evidence files for a report
   * 
   * @param reportId - Report ID
   * @returns Array of evidence files
   */
  static async getEvidence(reportId: number): Promise<EvidenceFile[]> {
    const query = `
      SELECT id, report_id, filename, original_filename, file_path,
             file_size, mime_type, uploaded_at
      FROM scam_evidence_files
      WHERE report_id = ?
      ORDER BY uploaded_at DESC
    `;

    try {
      const results = await executeQuery<RowDataPacket[]>(query, [reportId]);
      return results as EvidenceFile[];
    } catch (error) {
      console.error('[GeneralScamReportModel] Error fetching evidence:', error);
      throw new Error('Failed to fetch evidence files');
    }
  }
}
