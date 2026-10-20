/**
 * Centralized API client service for communicating with the FastAPI backend.
 */

import {
  ApiClientError,
  BatchPredictRequest,
  BatchPredictResponse,
  HealthResponse,
  PredictRequest,
  PredictResponse,
  RootResponse,
} from '../types/api';

/**
 * Resolve the backend API base URL from Vite environment variables.
 * Trailing slashes are safely stripped.
 */
export function getApiBaseUrl(): string {
  const envUrl = import.meta.env.VITE_API_BASE_URL;
  const baseUrl = (typeof envUrl === 'string' && envUrl.trim())
    ? envUrl.trim()
    : 'http://127.0.0.1:8000';
  return baseUrl.replace(/\/+$/, '');
}

const DEFAULT_TIMEOUT_MS = 15000;

/**
 * Internal helper for typed fetch requests with timeout and robust error formatting.
 */
async function request<T>(
  endpoint: string,
  options: RequestInit = {},
  timeoutMs: number = DEFAULT_TIMEOUT_MS
): Promise<T> {
  const baseUrl = getApiBaseUrl();
  const url = `${baseUrl}${endpoint.startsWith('/') ? '' : '/'}${endpoint}`;

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetch(url, {
      ...options,
      signal: controller.signal,
      headers: {
        'Accept': 'application/json',
        'Content-Type': 'application/json',
        ...options.headers,
      },
    });

    clearTimeout(timeoutId);

    // Parse JSON or extract plain text if JSON parsing fails
    let data: unknown;
    const contentType = response.headers.get('content-type') || '';
    if (contentType.includes('application/json')) {
      try {
        data = await response.json();
      } catch {
        data = null;
      }
    } else {
      const text = await response.text();
      data = { detail: text };
    }

    if (!response.ok) {
      let detailMessage = `Request failed with HTTP status ${response.status}`;

      if (data && typeof data === 'object') {
        const payload = data as Record<string, unknown>;
        if (typeof payload.detail === 'string') {
          detailMessage = payload.detail;
        } else if (Array.isArray(payload.detail)) {
          // FastAPI / Pydantic validation error array
          const messages = payload.detail.map((err: unknown) => {
            if (err && typeof err === 'object' && 'msg' in err) {
              const e = err as { loc?: string[]; msg: string };
              const field = e.loc ? e.loc.join('.') : 'field';
              return `${field}: ${e.msg}`;
            }
            return String(err);
          });
          detailMessage = messages.join(' | ');
        }
      }

      const error: ApiClientError = {
        message: detailMessage,
        status: response.status,
        detail: typeof data === 'string' ? data : JSON.stringify(data),
      };
      throw error;
    }

    return data as T;
  } catch (err: unknown) {
    clearTimeout(timeoutId);

    if (err && typeof err === 'object' && 'message' in err && 'status' in err) {
      // Re-throw our structured ApiClientError
      throw err;
    }

    if (err instanceof DOMException && err.name === 'AbortError') {
      const timeoutError: ApiClientError = {
        message: `Request timed out after ${timeoutMs / 1000} seconds. Please verify the backend is running and reachable.`,
      };
      throw timeoutError;
    }

    const networkError: ApiClientError = {
      message:
        err instanceof Error
          ? `Network error: ${err.message}. Is the backend service active at ${baseUrl}?`
          : 'Failed to establish connection with the inference API.',
    };
    throw networkError;
  }
}

/**
 * Fetch root metadata and service discovery details.
 */
export async function fetchRootInfo(): Promise<RootResponse> {
  return request<RootResponse>('/');
}

/**
 * Fetch backend health status and model readiness.
 */
export async function fetchHealth(): Promise<HealthResponse> {
  return request<HealthResponse>('/health');
}

/**
 * Predict sentiment for a single input text.
 */
export async function predictSentiment(text: string): Promise<PredictResponse> {
  const payload: PredictRequest = { text };
  return request<PredictResponse>('/predict', {
    method: 'POST',
    body: JSON.stringify(payload),
  });
}

/**
 * Predict sentiment for a batch of texts (up to 32 items).
 */
export async function predictBatch(texts: string[]): Promise<BatchPredictResponse> {
  const payload: BatchPredictRequest = { texts };
  return request<BatchPredictResponse>('/predict/batch', {
    method: 'POST',
    body: JSON.stringify(payload),
  });
}
