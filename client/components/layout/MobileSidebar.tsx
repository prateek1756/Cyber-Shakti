import { useState } from "react";
import { Link, useLocation } from "react-router-dom";
import { Shield, Menu, X, Scan, BookOpen, AlertTriangle, Grid3X3, Home } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

const navItems = [
  { href: "/", label: "Home", icon: Home },
  { href: "/features", label: "Features", icon: Grid3X3 },
  { href: "/phishing-scanner", label: "Scanner", icon: Scan },
  { href: "/tips", label: "Tips", icon: BookOpen },
  { href: "/alerts", label: "Alerts", icon: AlertTriangle },
];

export default function MobileSidebar() {
  const [isOpen, setIsOpen] = useState(false);
  const location = useLocation();

  const toggleSidebar = () => setIsOpen(!isOpen);
  const closeSidebar = () => setIsOpen(false);

  return (
    <div className="md:hidden">
      {/* Menu Button */}
      <Button variant="ghost" size="sm" onClick={toggleSidebar}>
        <Menu className="h-5 w-5" />
      </Button>

      {/* Sidebar Overlay */}
      {isOpen && (
        <div className="fixed inset-0 z-50">
          {/* Background Overlay */}
          <div 
            className="absolute inset-0 bg-black/50" 
            onClick={closeSidebar}
          />
          
          {/* Sidebar Panel */}
          <div className="absolute top-0 left-0 h-full w-72 bg-background/95 backdrop-blur-md border-r shadow-xl">
            {/* Header */}
            <div className="flex items-center justify-between p-4 border-b bg-background/90">
              <div className="flex items-center gap-2">
                <Shield className="h-6 w-6 text-primary" />
                <span className="font-bold text-primary">CyberShakti</span>
              </div>
              <Button variant="ghost" size="sm" onClick={closeSidebar}>
                <X className="h-5 w-5" />
              </Button>
            </div>

            {/* Navigation Links Only */}
            <div className="p-4 space-y-2 bg-background/80">
              {navItems.map((item) => {
                const Icon = item.icon;
                const isActive = location.pathname === item.href;
                
                return (
                  <Link
                    key={item.href}
                    to={item.href}
                    onClick={closeSidebar}
                    className={cn(
                      "flex items-center gap-3 p-3 rounded-lg transition-colors bg-background/90 backdrop-blur-sm",
                      isActive 
                        ? "bg-primary text-primary-foreground shadow-sm" 
                        : "hover:bg-muted/90 text-foreground"
                    )}
                  >
                    <Icon className="h-5 w-5" />
                    <span className="text-base font-medium">{item.label}</span>
                  </Link>
                );
              })}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}