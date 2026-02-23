/**
 * ScamMap Component
 * 
 * Interactive map showing user location and nearby scam reports
 * Uses Leaflet.js for map visualization
 */

import { useEffect, useRef } from 'react';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';

// Fix Leaflet default icon issue with bundlers
delete (L.Icon.Default.prototype as any)._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-icon-2x.png',
  iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-icon.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-shadow.png',
});

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

interface ScamMapProps {
  userLocation: UserLocation;
  scams: ScamReport[];
  radius: number;
}

export function ScamMap({ userLocation, scams, radius }: ScamMapProps) {
  const mapRef = useRef<L.Map | null>(null);
  const mapContainerRef = useRef<HTMLDivElement>(null);
  const markersRef = useRef<L.Marker[]>([]);
  const circleRef = useRef<L.Circle | null>(null);

  useEffect(() => {
    if (!mapContainerRef.current) return;

    // Initialize map
    if (!mapRef.current) {
      mapRef.current = L.map(mapContainerRef.current).setView(
        [userLocation.latitude, userLocation.longitude],
        12
      );

      // Add tile layer
      L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '© OpenStreetMap contributors',
        maxZoom: 19,
      }).addTo(mapRef.current);
    }

    // Clear existing markers and circle
    markersRef.current.forEach(marker => marker.remove());
    markersRef.current = [];
    if (circleRef.current) {
      circleRef.current.remove();
    }

    // Add user location marker (blue)
    const userIcon = L.divIcon({
      className: 'custom-user-marker',
      html: `
        <div style="
          width: 24px;
          height: 24px;
          background: #3b82f6;
          border: 3px solid white;
          border-radius: 50%;
          box-shadow: 0 2px 8px rgba(0,0,0,0.3);
        "></div>
      `,
      iconSize: [24, 24],
      iconAnchor: [12, 12],
    });

    const userMarker = L.marker(
      [userLocation.latitude, userLocation.longitude],
      { icon: userIcon }
    ).addTo(mapRef.current);

    userMarker.bindPopup('<b>Your Location</b>');
    markersRef.current.push(userMarker);

    // Add radius circle
    circleRef.current = L.circle(
      [userLocation.latitude, userLocation.longitude],
      {
        radius: radius * 1000, // Convert km to meters
        color: '#3b82f6',
        fillColor: '#3b82f6',
        fillOpacity: 0.1,
        weight: 2,
      }
    ).addTo(mapRef.current);

    // Add scam markers (red)
    scams.forEach(scam => {
      const scamIcon = L.divIcon({
        className: 'custom-scam-marker',
        html: `
          <div style="
            width: 20px;
            height: 20px;
            background: #ef4444;
            border: 2px solid white;
            border-radius: 50%;
            box-shadow: 0 2px 6px rgba(0,0,0,0.3);
          "></div>
        `,
        iconSize: [20, 20],
        iconAnchor: [10, 10],
      });

      const marker = L.marker([scam.latitude, scam.longitude], {
        icon: scamIcon,
      }).addTo(mapRef.current!);

      // Format scam type for display
      const formattedType = scam.scam_type
        .split('_')
        .map(word => word.charAt(0).toUpperCase() + word.slice(1))
        .join(' ');

      // Format date
      const date = new Date(scam.created_at);
      const formattedDate = date.toLocaleDateString();

      marker.bindPopup(`
        <div style="min-width: 200px;">
          <h3 style="font-weight: bold; margin-bottom: 8px;">${scam.title}</h3>
          <p style="font-size: 14px; color: #666; margin-bottom: 8px;">${scam.description}</p>
          <div style="display: flex; flex-direction: column; gap: 4px; font-size: 12px;">
            <span style="background: #fee2e2; color: #991b1b; padding: 2px 8px; border-radius: 12px; display: inline-block; width: fit-content;">
              ${formattedType}
            </span>
            <span style="color: #999;">
              ${scam.distance_km.toFixed(1)} km away • ${formattedDate}
            </span>
          </div>
        </div>
      `);

      markersRef.current.push(marker);
    });

    // Fit bounds to show all markers
    if (scams.length > 0) {
      const bounds = L.latLngBounds([
        [userLocation.latitude, userLocation.longitude],
        ...scams.map(s => [s.latitude, s.longitude] as [number, number]),
      ]);
      mapRef.current.fitBounds(bounds, { padding: [50, 50] });
    } else {
      mapRef.current.setView(
        [userLocation.latitude, userLocation.longitude],
        12
      );
    }

    // Cleanup
    return () => {
      // Don't destroy map on every render, just clean markers
    };
  }, [userLocation, scams, radius]);

  return (
    <div
      ref={mapContainerRef}
      className="w-full h-[500px] rounded-lg overflow-hidden border"
      style={{ zIndex: 0 }}
    />
  );
}
