import "./global.css";

import { Toaster } from "@/components/ui/toaster";
import { createRoot } from "react-dom/client";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import { lazy, Suspense } from "react";
import Index from "./pages/Index";
import SiteHeader from "@/components/layout/SiteHeader";
import SiteFooter from "@/components/layout/SiteFooter";
import PageIntro from "@/components/motion/PageIntro";
import ScrollProgress from "@/components/motion/ScrollProgress";

// Lazy load pages
const NotFound = lazy(() => import("./pages/NotFound"));
const Features = lazy(() => import("./pages/Features"));
const Tips = lazy(() => import("./pages/Tips"));
const Alerts = lazy(() => import("./pages/Alerts"));
const PhishingScanner = lazy(() => import("./pages/PhishingScanner"));
const FraudDetection = lazy(() => import("./pages/FraudDetection"));
const ScamCallBlocking = lazy(() => import("./pages/ScamCallBlocking"));
const FakeProfileVerification = lazy(() => import("./pages/FakeProfileVerification"));
const DeepfakeDetection = lazy(() => import("./pages/DeepfakeDetection"));

const queryClient = new QueryClient();

const App = () => (
  <QueryClientProvider client={queryClient}>
    <TooltipProvider>
      <Toaster />
      <Sonner />
      <div className="dark min-h-screen bg-background text-foreground">
        <BrowserRouter>
          <PageIntro />
          <ScrollProgress />
          <SiteHeader />
          <main className="min-h-[calc(100vh-8rem)]">
            <Suspense fallback={<div className="flex items-center justify-center min-h-[50vh]"><div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div></div>}>
              <Routes>
                <Route path="/" element={<Index />} />
                <Route path="/features" element={<Features />} />
                <Route path="/phishing-scanner" element={<PhishingScanner />} />
                <Route path="/fraud-detection" element={<FraudDetection />} />
                <Route path="/scam-call-blocking" element={<ScamCallBlocking />} />
                <Route path="/fake-profile-verification" element={<FakeProfileVerification />} />
                <Route path="/deepfake-detection" element={<DeepfakeDetection />} />
                <Route path="/tips" element={<Tips />} />
                <Route path="/alerts" element={<Alerts />} />
                <Route path="*" element={<NotFound />} />
              </Routes>
            </Suspense>
          </main>
          <SiteFooter />
        </BrowserRouter>
      </div>
    </TooltipProvider>
  </QueryClientProvider>
);

createRoot(document.getElementById("root")!).render(<App />);
