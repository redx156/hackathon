/**
 * API Service for Pneumonia Detection Backend
 * 
 * Handles all communication with FastAPI backend
 * Uses VITE_API_BASE_URL env var for Docker compatibility
 */

// Read API URL from environment (set via Docker or .env)
// Fallback to localhost:8000 for local development
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';
const API_TIMEOUT_MS = 30000; // 30 seconds timeout

/**
 * Custom error class for API errors
 */
export class APIError extends Error {
    constructor(message, status = null, details = null) {
        super(message);
        this.name = 'APIError';
        this.status = status;
        this.details = details;
    }
}

/**
 * Analyze X-ray image for pneumonia detection
 * 
 * @param {File} file - Image file (PNG/JPEG)
 * @returns {Promise<Object>} API response with prediction, confidence, heatmap, etc.
 * @throws {APIError} On network, timeout, or server errors
 */
export async function analyzeXRay(file) {
    // Validate input
    if (!file) {
        throw new APIError('No file provided', null, 'empty_upload');
    }

    if (!file.type.startsWith('image/')) {
        throw new APIError('Invalid file type. Please upload an image file.', null, 'invalid_type');
    }

    // Create FormData for multipart upload
    const formData = new FormData();
    formData.append('file', file);

    // Create AbortController for timeout
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), API_TIMEOUT_MS);

    try {
        const response = await fetch(`${API_BASE_URL}/api/predict`, {
            method: 'POST',
            body: formData,
            signal: controller.signal,
        });

        clearTimeout(timeoutId);

        // Handle HTTP errors
        if (!response.ok) {
            let errorMessage = 'Server error';
            let errorDetails = null;

            try {
                const errorData = await response.json();
                errorMessage = errorData.detail || errorMessage;
                errorDetails = errorData;
            } catch {
                // Response wasn't JSON
            }

            if (response.status === 400) {
                throw new APIError('Invalid image file. Please upload a valid X-ray image.', 400, 'invalid_image');
            } else if (response.status === 500) {
                throw new APIError('Server error during analysis. Please try again.', 500, 'server_error');
            } else {
                throw new APIError(errorMessage, response.status, errorDetails);
            }
        }

        // Parse successful response
        const data = await response.json();

        // Validate response shape
        if (!data.prediction || typeof data.confidence !== 'number') {
            throw new APIError('Invalid response from server', null, 'invalid_response');
        }

        return {
            prediction: data.prediction,
            confidence: data.confidence,
            confidencePercent: Math.round(data.confidence * 100),
            riskLevel: data.risk_level,
            heatmapDataUri: data.heatmap_data_uri,
            heatmapImage: data.heatmap_image,
            lowImageQuality: data.low_image_quality || false,
            note: data.note || '',
            device: data.device,
        };

    } catch (error) {
        clearTimeout(timeoutId);

        // Handle abort (timeout)
        if (error.name === 'AbortError') {
            throw new APIError('Request timed out. Please check your connection and try again.', null, 'timeout');
        }

        // Handle network errors
        if (error.name === 'TypeError' && error.message.includes('fetch')) {
            throw new APIError('Cannot connect to server. Is the backend running?', null, 'network_error');
        }

        // Re-throw APIError as-is
        if (error instanceof APIError) {
            throw error;
        }

        // Unknown error
        throw new APIError(`Unexpected error: ${error.message}`, null, 'unknown');
    }
}

/**
 * Check if the backend is healthy
 * 
 * @returns {Promise<Object>} Health status
 */
export async function checkHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/health`, {
            method: 'GET',
        });

        if (!response.ok) {
            return { healthy: false, error: 'Server returned error' };
        }

        const data = await response.json();
        return {
            healthy: data.status === 'healthy',
            modelLoaded: data.model_loaded,
            device: data.device,
        };

    } catch (error) {
        return { healthy: false, error: error.message };
    }
}
