/**
 * ScamReportForm Component
 * 
 * Form for submitting new scam reports
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
import { X, Loader2 } from 'lucide-react';
import { useToast } from '@/hooks/use-toast';

interface UserLocation {
  latitude: number;
  longitude: number;
}

interface ScamReportFormProps {
  userLocation: UserLocation;
  onClose: () => void;
  onSubmit: () => void;
}

export function ScamReportForm({ userLocation, onClose, onSubmit }: ScamReportFormProps) {
  const [title, setTitle] = useState('');
  const [description, setDescription] = useState('');
  const [scamType, setScamType] = useState<string>('other');
  const [loading, setLoading] = useState(false);
  const { toast } = useToast();

  const scamTypes = [
    { value: 'phishing', label: 'Phishing' },
    { value: 'phone_scam', label: 'Phone Scam' },
    { value: 'fake_website', label: 'Fake Website' },
    { value: 'identity_theft', label: 'Identity Theft' },
    { value: 'investment_fraud', label: 'Investment Fraud' },
    { value: 'romance_scam', label: 'Romance Scam' },
    { value: 'tech_support', label: 'Tech Support' },
    { value: 'other', label: 'Other' },
  ];

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    // Validation
    if (title.length < 5) {
      toast({
        title: 'Validation Error',
        description: 'Title must be at least 5 characters',
        variant: 'destructive',
      });
      return;
    }

    if (description.length < 10) {
      toast({
        title: 'Validation Error',
        description: 'Description must be at least 10 characters',
        variant: 'destructive',
      });
      return;
    }

    setLoading(true);

    try {
      const response = await fetch('/api/scams/report', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          title,
          description,
          scam_type: scamType,
          latitude: userLocation.latitude,
          longitude: userLocation.longitude,
        }),
      });

      const data = await response.json();

      if (!response.ok || !data.success) {
        throw new Error(data.error || 'Failed to submit report');
      }

      toast({
        title: 'Success',
        description: data.data.message,
      });

      onSubmit();
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

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4">
      <Card className="w-full max-w-2xl max-h-[90vh] overflow-y-auto">
        <CardHeader>
          <div className="flex items-start justify-between">
            <div>
              <CardTitle>Report a Scam</CardTitle>
              <CardDescription>
                Help protect your community by reporting scams
              </CardDescription>
            </div>
            <Button
              variant="ghost"
              size="icon"
              onClick={onClose}
              disabled={loading}
            >
              <X className="h-4 w-4" />
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSubmit} className="space-y-4">
            {/* Title */}
            <div className="space-y-2">
              <label htmlFor="title" className="text-sm font-medium">
                Title <span className="text-destructive">*</span>
              </label>
              <input
                id="title"
                type="text"
                value={title}
                onChange={(e) => setTitle(e.target.value)}
                placeholder="Brief description of the scam"
                className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
                required
                minLength={5}
                maxLength={255}
                disabled={loading}
              />
              <p className="text-xs text-muted-foreground">
                {title.length}/255 characters (minimum 5)
              </p>
            </div>

            {/* Scam Type */}
            <div className="space-y-2">
              <label htmlFor="scamType" className="text-sm font-medium">
                Scam Type <span className="text-destructive">*</span>
              </label>
              <select
                id="scamType"
                value={scamType}
                onChange={(e) => setScamType(e.target.value)}
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

            {/* Description */}
            <div className="space-y-2">
              <label htmlFor="description" className="text-sm font-medium">
                Description <span className="text-destructive">*</span>
              </label>
              <textarea
                id="description"
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                placeholder="Provide detailed information about the scam..."
                className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm min-h-[120px]"
                required
                minLength={10}
                maxLength={2000}
                disabled={loading}
              />
              <p className="text-xs text-muted-foreground">
                {description.length}/2000 characters (minimum 10)
              </p>
            </div>

            {/* Location Info */}
            <div className="rounded-lg bg-muted p-3 text-sm">
              <p className="font-medium mb-1">Location</p>
              <p className="text-muted-foreground text-xs">
                Latitude: {userLocation.latitude.toFixed(4)}, Longitude:{' '}
                {userLocation.longitude.toFixed(4)}
              </p>
            </div>

            {/* Submit Buttons */}
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
              <Button
                type="button"
                variant="outline"
                onClick={onClose}
                disabled={loading}
              >
                Cancel
              </Button>
            </div>

            <p className="text-xs text-muted-foreground">
              Your report will be reviewed by our team before being published.
            </p>
          </form>
        </CardContent>
      </Card>
    </div>
  );
}
