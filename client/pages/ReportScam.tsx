/**
 * Report Scam Page
 * 
 * General scam reporting form (non-location-based)
 */

import { useState } from 'react';
import { Button } from '@/components/ui/button';
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import { AlertTriangle, Upload, Loader2, CheckCircle } from 'lucide-react';
import { useToast } from '@/hooks/use-toast';

export default function ReportScam() {
  const [formData, setFormData] = useState({
    title: '',
    description: '',
    scam_type: 'other',
    severity_level: 'medium',
    reporter_name: '',
    reporter_email: '',
    reporter_phone: '',
  });

  const [files, setFiles] = useState<FileList | null>(null);
  const [loading, setLoading] = useState(false);
  const [submitted, setSubmitted] = useState(false);
  const { toast } = useToast();

  const scamTypes = [
    { value: 'phishing', label: 'Phishing / Email Scam' },
    { value: 'phone_scam', label: 'Phone / Call Scam' },
    { value: 'fake_website', label: 'Fake Website' },
    { value: 'identity_theft', label: 'Identity Theft' },
    { value: 'investment_fraud', label: 'Investment Fraud' },
    { value: 'romance_scam', label: 'Romance Scam' },
    { value: 'tech_support', label: 'Tech Support Scam' },
    { value: 'online_shopping', label: 'Online Shopping Scam' },
    { value: 'job_scam', label: 'Job / Employment Scam' },
    { value: 'lottery_scam', label: 'Lottery / Prize Scam' },
    { value: 'charity_scam', label: 'Charity / Donation Scam' },
    { value: 'other', label: 'Other' },
  ];

  const severityLevels = [
    { value: 'low', label: 'Low', description: 'Minor inconvenience, no financial loss' },
    { value: 'medium', label: 'Medium', description: 'Potential financial loss or data exposure' },
    { value: 'high', label: 'High', description: 'Significant financial loss or identity theft' },
    { value: 'critical', label: 'Critical', description: 'Severe financial loss or widespread impact' },
  ];

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    // Validation
    if (formData.title.length < 5) {
      toast({
        title: 'Validation Error',
        description: 'Title must be at least 5 characters',
        variant: 'destructive',
      });
      return;
    }

    if (formData.description.length < 20) {
      toast({
        title: 'Validation Error',
        description: 'Description must be at least 20 characters',
        variant: 'destructive',
      });
      return;
    }

    setLoading(true);

    try {
      // Create FormData for file upload
      const submitData = new FormData();
      submitData.append('title', formData.title);
      submitData.append('description', formData.description);
      submitData.append('scam_type', formData.scam_type);
      submitData.append('severity_level', formData.severity_level);
      
      if (formData.reporter_name) submitData.append('reporter_name', formData.reporter_name);
      if (formData.reporter_email) submitData.append('reporter_email', formData.reporter_email);
      if (formData.reporter_phone) submitData.append('reporter_phone', formData.reporter_phone);

      // Add files
      if (files) {
        Array.from(files).forEach((file) => {
          submitData.append('evidence', file);
        });
      }

      const response = await fetch('/api/reports/submit', {
        method: 'POST',
        body: submitData,
      });

      const data = await response.json();

      if (!response.ok || !data.success) {
        throw new Error(data.error || 'Failed to submit report');
      }

      toast({
        title: 'Success',
        description: data.data.message,
      });

      setSubmitted(true);
    } catch (error) {
      console.error('Error submitting report:', error);
      toast({
        title: 'Error',
        description: error instanceof Error ? error.message : 'Failed to submit report',
        variant: 'destructive',
      });
    } finally {
      setLoading(false);
    }
  };

  if (submitted) {
    return (
      <div className="container py-8">
        <Card className="max-w-2xl mx-auto">
          <CardContent className="py-12 text-center">
            <div className="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-full bg-primary/10">
              <CheckCircle className="h-8 w-8 text-primary" />
            </div>
            <h2 className="mb-2 text-2xl font-bold">Report Submitted Successfully!</h2>
            <p className="text-muted-foreground mb-6">
              Thank you for helping keep our community safe. Our team will review your report shortly.
            </p>
            <Button onClick={() => setSubmitted(false)}>
              Submit Another Report
            </Button>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="container py-8 space-y-6">
      {/* Header */}
      <div className="text-center space-y-2">
        <h1 className="text-4xl font-extrabold tracking-tight md:text-5xl">
          Report a Scam
        </h1>
        <p className="mx-auto max-w-2xl text-muted-foreground">
          Help protect others by reporting scams you've encountered
        </p>
      </div>

      {/* Form */}
      <Card className="max-w-3xl mx-auto">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <AlertTriangle className="h-5 w-5 text-primary" />
            Scam Report Form
          </CardTitle>
          <CardDescription>
            All fields marked with * are required. Your contact information is optional.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSubmit} className="space-y-6">
            {/* Title */}
            <div className="space-y-2">
              <label htmlFor="title" className="text-sm font-medium">
                Scam Title <span className="text-destructive">*</span>
              </label>
              <input
                id="title"
                type="text"
                value={formData.title}
                onChange={(e) => setFormData({ ...formData, title: e.target.value })}
                placeholder="Brief description of the scam"
                className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
                required
                minLength={5}
                maxLength={255}
                disabled={loading}
              />
              <p className="text-xs text-muted-foreground">
                {formData.title.length}/255 characters (minimum 5)
              </p>
            </div>

            {/* Scam Type */}
            <div className="space-y-2">
              <label htmlFor="scam_type" className="text-sm font-medium">
                Scam Type <span className="text-destructive">*</span>
              </label>
              <select
                id="scam_type"
                value={formData.scam_type}
                onChange={(e) => setFormData({ ...formData, scam_type: e.target.value })}
                className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
                required
                disabled={loading}
              >
                {scamTypes.map((type) => (
                  <option key={type.value} value={type.value}>
                    {type.label}
                  </option>
                ))}
              </select>
            </div>

            {/* Severity Level */}
            <div className="space-y-2">
              <label className="text-sm font-medium">
                Severity Level <span className="text-destructive">*</span>
              </label>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                {severityLevels.map((level) => (
                  <label
                    key={level.value}
                    className={`flex items-start gap-3 rounded-lg border p-3 cursor-pointer transition-colors ${
                      formData.severity_level === level.value
                        ? 'border-primary bg-primary/5'
                        : 'border-input hover:bg-accent'
                    }`}
                  >
                    <input
                      type="radio"
                      name="severity_level"
                      value={level.value}
                      checked={formData.severity_level === level.value}
                      onChange={(e) => setFormData({ ...formData, severity_level: e.target.value })}
                      className="mt-1"
                      disabled={loading}
                    />
                    <div className="flex-1">
                      <div className="font-medium">{level.label}</div>
                      <div className="text-xs text-muted-foreground">{level.description}</div>
                    </div>
                  </label>
                ))}
              </div>
            </div>

            {/* Description */}
            <div className="space-y-2">
              <label htmlFor="description" className="text-sm font-medium">
                Detailed Description <span className="text-destructive">*</span>
              </label>
              <textarea
                id="description"
                value={formData.description}
                onChange={(e) => setFormData({ ...formData, description: e.target.value })}
                placeholder="Provide detailed information about the scam, including how it happened, what was said, any URLs or phone numbers involved, etc."
                className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm min-h-[150px]"
                required
                minLength={20}
                maxLength={5000}
                disabled={loading}
              />
              <p className="text-xs text-muted-foreground">
                {formData.description.length}/5000 characters (minimum 20)
              </p>
            </div>

            {/* Evidence Upload */}
            <div className="space-y-2">
              <label htmlFor="evidence" className="text-sm font-medium">
                Evidence Files (Optional)
              </label>
              <div className="flex items-center gap-2">
                <label
                  htmlFor="evidence"
                  className="flex items-center gap-2 rounded-md border border-input bg-background px-4 py-2 text-sm cursor-pointer hover:bg-accent"
                >
                  <Upload className="h-4 w-4" />
                  Choose Files
                </label>
                <input
                  id="evidence"
                  type="file"
                  multiple
                  accept="image/*,.pdf,.txt,.doc,.docx"
                  onChange={(e) => setFiles(e.target.files)}
                  className="hidden"
                  disabled={loading}
                />
                <span className="text-xs text-muted-foreground">
                  {files ? `${files.length} file(s) selected` : 'No files selected'}
                </span>
              </div>
              <p className="text-xs text-muted-foreground">
                Upload screenshots, emails, or documents (max 5 files, 5MB each)
              </p>
            </div>

            {/* Reporter Information (Optional) */}
            <div className="space-y-4 rounded-lg border p-4">
              <h3 className="font-medium">Your Contact Information (Optional)</h3>
              <p className="text-xs text-muted-foreground">
                Providing your contact information helps us follow up if needed
              </p>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="space-y-2">
                  <label htmlFor="reporter_name" className="text-sm font-medium">
                    Name
                  </label>
                  <input
                    id="reporter_name"
                    type="text"
                    value={formData.reporter_name}
                    onChange={(e) => setFormData({ ...formData, reporter_name: e.target.value })}
                    placeholder="Your name"
                    className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
                    maxLength={100}
                    disabled={loading}
                  />
                </div>

                <div className="space-y-2">
                  <label htmlFor="reporter_email" className="text-sm font-medium">
                    Email
                  </label>
                  <input
                    id="reporter_email"
                    type="email"
                    value={formData.reporter_email}
                    onChange={(e) => setFormData({ ...formData, reporter_email: e.target.value })}
                    placeholder="your@email.com"
                    className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
                    maxLength={255}
                    disabled={loading}
                  />
                </div>

                <div className="space-y-2">
                  <label htmlFor="reporter_phone" className="text-sm font-medium">
                    Phone
                  </label>
                  <input
                    id="reporter_phone"
                    type="tel"
                    value={formData.reporter_phone}
                    onChange={(e) => setFormData({ ...formData, reporter_phone: e.target.value })}
                    placeholder="+1234567890"
                    className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
                    maxLength={20}
                    disabled={loading}
                  />
                </div>
              </div>
            </div>

            {/* Submit Button */}
            <div className="flex gap-2 pt-2">
              <Button
                type="submit"
                className="flex-1"
                disabled={loading}
              >
                {loading ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Submitting...
                  </>
                ) : (
                  'Submit Report'
                )}
              </Button>
            </div>

            <p className="text-xs text-muted-foreground text-center">
              Your report will be reviewed by our team before being published.
            </p>
          </form>
        </CardContent>
      </Card>
    </div>
  );
}
