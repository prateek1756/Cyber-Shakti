import { createServer } from '../server/index';

console.log("[Vercel] Function api/index.ts starting up... Version 1.0.1");
// Create the app without the PythonBridge for serverless
// The Python services will be handled as separate serverless functions
const app = createServer();

export default app;
