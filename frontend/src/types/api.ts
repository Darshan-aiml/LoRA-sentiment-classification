/**
 * Type definitions matching the FastAPI backend schemas exactly.
 */

export interface RootResponse {
  name: string;
  version: string;
  task: string;
  endpoints: Record<string, string>;
}

export type HealthStatus = 'ready' | 'degraded' | 'initializing' | 'unavailable';

export interface HealthResponse {
  status: HealthStatus;
  model_loaded: boolean;
  device: string;
  base_model: string;
  adapter_path: string;
  lora_rank?: number | null;
  details?: string | null;
}

export interface SentimentProbabilities {
  negative: number;
  positive: number;
}

export interface PredictRequest {
  text: string;
}

export interface PredictResponse {
  text: string;
  sentiment: 'positive' | 'negative';
  confidence: number;
  probabilities: SentimentProbabilities;
  model: string;
  adapter: string;
  rank: number;
}

export interface BatchPredictRequest {
  texts: string[];
}

export interface BatchPredictResponse {
  predictions: PredictResponse[];
  count: number;
}

export interface ApiClientError {
  message: string;
  status?: number;
  detail?: string;
}
