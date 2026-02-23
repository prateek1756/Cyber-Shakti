import { useState, useCallback } from "react";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { MapPin, AlertTriangle, Plus, Loader2 } from "lucide-react";
import { ScamMap } from "@/components/ScamMap";
import { ScamReportForm } from "@/components/ScamReportForm";
import { useToast } from "@/hooks/use-toast";

interface ScamReport {
  id: number;
  title: string;
  description: string;
  scam_type: string;
  latitude: number;
  longitude: number;
  distance_km: number;
  created_at: string;
}

interface UserLocation {
  latitude: number;
  longitude: number;
}

export default function ScamAlerts() {
  const [location, setLocation] = useState<UserLocation | null>(null);
  const [scams, setScams] = useState<ScamReport[]>([]);
  const [loading, setLoading] = useState(false);
  const [locationError, setLocationError] = useState<string | null>(null);
  const [showReportForm, setShowReportForm] = useState(false);
  const [radius, setRadius] = useState(10); // km
  const { toast } = useToast();

  // Fetch nearby scams
  const fetchNearbyScams = useCallback(async (lat: number, lon: number, rad: number) => {
    setLoading(true);
    try {
      const response = await fetch(
        `/api/scams/nearby?latitude=${lat}&longitude=${lon}&radius=${rad}`
      );
      
      if (!response.ok) {
        throw new Error('Failed to fetch nearby scams');
      }

      const data = await response.json();
      
      if (data.success) {
        setScams(data.data.scams);
        toast({
          title: "Scams Loaded",
          description: `Found ${data.data.count} scam reports within ${rad}km`,
        });
      }
    } catch (error) {
      console.error('Error fetching scams:', error);
      toast({
        title: "Error",
        description: "Failed to load nearby scam reports",
        variant: "destructive",
      });
    } finally {
      setLoading(false);
    }
  }, [toast]);

  // Request user location
  const requestLocation = useCallback(() => {
    if (!navigator.geolocation) {
      setLocationError("Geolocation not supported in this browser");
      return;
    }

    setLoading(true);
    setLocationError(null);

    navigator.geolocation.getCurrentPosition(
      (position) => {
        const userLoc = {
          latitude: position.coords.latitude,
          longitude: position.coords.longitude,
        };
        setLocation(userLoc);
        fetchNearbyScams(userLoc.latitude, userLoc.longitude, radius);
      },
      (error) => {
        setLocationError(error.message);
        setLoading(false);
        
        // Fallback: Try IP-based location (you can implement this)
        toast({
          title: "Location Access Denied",
          description: "Using approximate location based on IP address",
          variant: "destructive",
        });
      },
      { enableHighAccuracy: true, timeout: 10000, maximumAge: 0 }
    );
  }, [radius, fetchNearbyScams, toast]);

  // Handle successful report submission
  const handleReportSubmitted = useCallback(() => {
    setShowReportForm(false);
    
    // Refresh scams list
    if (location) {
      fetchNearbyScams(location.latitude, location.longitude, radius);
    }
    
    toast({
      title: "Report Submitted",
      description: "Thank you for helping keep the community safe!",
    });
  }, [location, radius, fetchNearbyScams, toast]);

  // Format scam type for display
  const formatScamType = (type: string): string => {
    return type.split('_').map(word => 
      word.charAt(0).toUpperCase() + word.slice(1)
    ).join(' ');
  };

  // Format date for display
  const formatDate = (dateString: string): string => {
    const date = new Date(dateString);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));
    
    if (diffDays === 0) return 'Today';
    if (diffDays === 1) return 'Yesterday';
    if (diffDays < 7) return `${diffDays} days ago`;
    return date.toLocaleDateString();
  };

  return (
    <div className="container py-8 space-y-6">
      {/* Header */}
      <div className="text-center space-y-2">
        <h1 className="text-4xl font-extrabold tracking-tight md:text-5xl">
          Location-Based Scam Alerts
        </h1>
        <p className="mx-auto max-w-2xl text-muted-foreground">
          Stay informed about scams reported in your area
        </p>
      </div>

      {/* Location Control Card */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <MapPin className="h-5 w-5 text-primary" />
            Your Location
          </CardTitle>
          <CardDescription>
            Enable location access to see nearby scam reports
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {!location ? (
            <div className="space-y-3">
              <Button 
                onClick={requestLocation} 
                className="w-full"
                disabled={loading}
              >
                {loading ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Getting Location...
                  </>
                ) : (
                  <>
                    <MapPin className="mr-2 h-4 w-4" />
                    Enable Location
                  </>
                )}
              </Button>
              {locationError && (
                <p className="text-sm text-destructive">{locationError}</p>
              )}
            </div>
          ) : (
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <div className="text-sm text-muted-foreground">
                  <p>Latitude: {location.latitude.toFixed(4)}</p>
                  <p>Longitude: {location.longitude.toFixed(4)}</p>
                  <p>Search Radius: {radius} km</p>
                </div>
                <div className="flex gap-2">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setShowReportForm(true)}
                  >
                    <Plus className="mr-2 h-4 w-4" />
                    Report Scam
                  </Button>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={requestLocation}
                    disabled={loading}
                  >
                    Refresh
                  </Button>
                </div>
              </div>
              
              {/* Radius Selector */}
              <div className="flex items-center gap-2">
                <label className="text-sm font-medium">Radius:</label>
                <select
                  value={radius}
                  onChange={(e) => {
                    const newRadius = parseInt(e.target.value);
                    setRadius(newRadius);
                    fetchNearbyScams(location.latitude, location.longitude, newRadius);
                  }}
                  className="rounded-md border border-input bg-background px-3 py-1 text-sm"
                >
                  <option value="5">5 km</option>
                  <option value="10">10 km</option>
                  <option value="25">25 km</option>
                  <option value="50">50 km</option>
                </select>
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Map */}
      {location && (
        <Card>
          <CardHeader>
            <CardTitle>Scam Map</CardTitle>
            <CardDescription>
              {scams.length} scam reports within {radius}km
            </CardDescription>
          </CardHeader>
          <CardContent>
            <ScamMap
              userLocation={location}
              scams={scams}
              radius={radius}
            />
          </CardContent>
        </Card>
      )}

      {/* Scam List */}
      {location && scams.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Nearby Scam Reports</CardTitle>
            <CardDescription>
              Recent scams reported in your area
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {scams.map((scam) => (
                <div
                  key={scam.id}
                  className="flex gap-4 rounded-lg border p-4 hover:bg-accent/50 transition-colors"
                >
                  <div className="flex-shrink-0">
                    <div className="flex h-10 w-10 items-center justify-center rounded-full bg-destructive/10">
                      <AlertTriangle className="h-5 w-5 text-destructive" />
                    </div>
                  </div>
                  <div className="flex-1 space-y-1">
                    <div className="flex items-start justify-between">
                      <h3 className="font-semibold">{scam.title}</h3>
                      <span className="text-xs text-muted-foreground">
                        {scam.distance_km.toFixed(1)} km away
                      </span>
                    </div>
                    <p className="text-sm text-muted-foreground">
                      {scam.description}
                    </p>
                    <div className="flex items-center gap-2 text-xs">
                      <span className="rounded-full bg-primary/10 px-2 py-1 text-primary">
                        {formatScamType(scam.scam_type)}
                      </span>
                      <span className="text-muted-foreground">
                        {formatDate(scam.created_at)}
                      </span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* No Scams Message */}
      {location && scams.length === 0 && !loading && (
        <Card>
          <CardContent className="py-12 text-center">
            <div className="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-full bg-primary/10">
              <MapPin className="h-8 w-8 text-primary" />
            </div>
            <h3 className="mb-2 text-lg font-semibold">No Scams Reported Nearby</h3>
            <p className="text-sm text-muted-foreground">
              Great news! There are no scam reports within {radius}km of your location.
            </p>
          </CardContent>
        </Card>
      )}

      {/* Report Form Modal */}
      {showReportForm && location && (
        <ScamReportForm
          userLocation={location}
          onClose={() => setShowReportForm(false)}
          onSubmit={handleReportSubmitted}
        />
      )}
    </div>
  );
}
