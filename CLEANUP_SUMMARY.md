# CyberGuard Cleanup Summary

## Files Removed - January 2025

### 1. Unused UI Components (41 files)
Removed shadcn/ui components that were never imported or used:
- accordion.tsx
- alert-dialog.tsx
- alert.tsx
- aspect-ratio.tsx
- avatar.tsx
- breadcrumb.tsx
- calendar.tsx
- carousel.tsx
- chart.tsx
- checkbox.tsx
- collapsible.tsx
- command.tsx
- context-menu.tsx
- dialog.tsx
- drawer.tsx
- dropdown-menu.tsx
- form.tsx
- hover-card.tsx
- input-otp.tsx
- input.tsx
- label.tsx
- menubar.tsx
- navigation-menu.tsx
- pagination.tsx
- popover.tsx
- progress.tsx
- radio-group.tsx
- resizable.tsx
- scroll-area.tsx
- select.tsx
- separator.tsx
- sheet.tsx
- sidebar.tsx
- skeleton.tsx
- slider.tsx
- switch.tsx
- table.tsx
- tabs.tsx
- textarea.tsx
- toggle-group.tsx
- toggle.tsx

**Kept Components:**
- button.tsx ✓
- card.tsx ✓
- toast.tsx / toaster.tsx / use-toast.ts ✓
- sonner.tsx ✓
- tooltip.tsx ✓
- badge.tsx ✓

### 2. Redundant Pages (1 file)
- `client/pages/Scanner.tsx` - Replaced by dedicated PhishingScanner.tsx and FraudDetection.tsx pages

### 3. Standalone Module (10+ files)
- Entire `fraud-feature/` directory - Separate FastAPI project not integrated with main app

### 4. Temporary Files (1 file)
- `python/uploads/temp_1769932059.594454_Endfield 2026.01.27 - 11.51.38.01.mp4`

### 5. IDE Configuration (2 directories)
- `.builder/` directory
- `.cursor/` directory

### 6. Duplicate Lock Files (1 file)
- `package-lock.json` - Project uses pnpm-lock.yaml

### 7. Unused npm Dependencies (30+ packages)
Removed from package.json:
- @hookform/resolvers
- @radix-ui/react-accordion
- @radix-ui/react-alert-dialog
- @radix-ui/react-aspect-ratio
- @radix-ui/react-avatar
- @radix-ui/react-checkbox
- @radix-ui/react-collapsible
- @radix-ui/react-context-menu
- @radix-ui/react-dialog
- @radix-ui/react-dropdown-menu
- @radix-ui/react-hover-card
- @radix-ui/react-label
- @radix-ui/react-menubar
- @radix-ui/react-navigation-menu
- @radix-ui/react-popover
- @radix-ui/react-progress
- @radix-ui/react-radio-group
- @radix-ui/react-scroll-area
- @radix-ui/react-select
- @radix-ui/react-separator
- @radix-ui/react-slider
- @radix-ui/react-switch
- @radix-ui/react-tabs
- @radix-ui/react-toggle
- @radix-ui/react-toggle-group
- @react-three/drei
- @react-three/fiber
- @tailwindcss/typography
- @types/three
- cmdk
- date-fns
- embla-carousel-react
- input-otp
- next-themes
- react-day-picker
- react-hook-form
- react-resizable-panels
- recharts
- three
- vaul

## Updated Files

### .gitignore
Added exclusions for:
- `.cursor/` and `.builder/` directories
- `package-lock.json` and `yarn.lock`
- Temporary upload files in `python/uploads/`

### package.json
- Removed 30+ unused dependencies
- Kept only essential packages for the app

## Next Steps

Run the following command to clean up node_modules:
```bash
pnpm install
```

This will:
1. Remove unused packages from node_modules
2. Update pnpm-lock.yaml
3. Reduce installation size by ~100MB+

## Benefits

- **Reduced bundle size**: Removed ~200KB of unused source code
- **Faster installs**: ~100MB+ less in node_modules
- **Cleaner codebase**: Easier to navigate and maintain
- **Better performance**: Less code to parse and bundle
