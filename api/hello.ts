import type { VercelRequest, VercelResponse } from '@vercel/node';

export default function handler(
    request: VercelRequest,
    response: VercelResponse,
) {
    response.status(200).json({
        message: 'Hello from Vercel Serverless!',
        timestamp: new Date().toISOString(),
    });
}
