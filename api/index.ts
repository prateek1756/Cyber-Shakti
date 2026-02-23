import { createServer } from '../server/index';

// Create the app without the PythonBridge for serverless
// The Python services will be handled as separate serverless functions
const app = createServer();

export default app;
