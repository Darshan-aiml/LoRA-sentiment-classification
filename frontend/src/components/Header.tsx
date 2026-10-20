import React from 'react';
import { BookOpen, Cpu, RefreshCw } from 'lucide-react';
import { HealthResponse } from '../types/api';
import { getApiBaseUrl } from '../services/api';

interface HeaderProps {
  health: HealthResponse | null;
  loadingHealth: boolean;
  onRefreshHealth: () => void;
}

export const Header: React.FC<HeaderProps> = ({
  health,
  loadingHealth,
  onRefreshHealth,
}) => {
  const baseUrl = getApiBaseUrl();
  const docsUrl = `${baseUrl}/docs`;

  const isReady = health?.status === 'ready' && health.model_loaded;
  const isInitializing = health?.status === 'initializing';

  return (
    <header className="border-b border-slate-800/80 bg-slate-900/60 backdrop-blur-md sticky top-0 z-40">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-3.5 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3">
        {/* Title and Tagline */}
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-lg bg-indigo-950/80 border border-indigo-700/40 flex items-center justify-center text-indigo-400 shadow-sm">
            <Cpu className="w-5 h-5 text-indigo-400" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-lg sm:text-xl font-semibold tracking-tight text-white">
                LoRA Sentiment AI
              </h1>
              <span className="text-[11px] font-mono px-2 py-0.5 rounded-full bg-indigo-950/70 border border-indigo-800/50 text-indigo-300 font-medium">
                PEFT r=16
              </span>
            </div>
            <p className="text-xs text-slate-400">
              Parameter-efficient sentiment classification
            </p>
          </div>
        </div>

        {/* Status and Action Buttons */}
        <div className="flex items-center gap-2.5 sm:gap-3 flex-wrap">
          {/* Health Status Pill */}
          <div
            className={`inline-flex items-center gap-2 px-3 py-1.5 rounded-md border text-xs font-mono transition-colors ${
              isReady
                ? 'bg-emerald-950/40 border-emerald-800/50 text-emerald-300'
                : isInitializing
                ? 'bg-amber-950/40 border-amber-800/50 text-amber-300'
                : 'bg-rose-950/40 border-rose-800/50 text-rose-300'
            }`}
            title={health?.details || (isReady ? 'Service online & ready' : 'Service offline')}
          >
            <span
              className={`w-2 h-2 rounded-full ${
                isReady
                  ? 'bg-emerald-400 shadow-[0_0_8px_rgba(52,211,153,0.6)]'
                  : isInitializing
                  ? 'bg-amber-400 animate-pulse'
                  : 'bg-rose-400'
              }`}
            />
            <span className="font-sans font-medium">
              {isReady
                ? `API Ready (${health.device.toUpperCase()})`
                : isInitializing
                ? 'Initializing...'
                : 'API Offline'}
            </span>
            <button
              onClick={onRefreshHealth}
              disabled={loadingHealth}
              aria-label="Refresh API health status"
              className="ml-1 text-slate-400 hover:text-white transition-transform active:rotate-180 disabled:opacity-50"
            >
              <RefreshCw
                className={`w-3 h-3 ${loadingHealth ? 'animate-spin' : ''}`}
              />
            </button>
          </div>

          {/* Interactive Docs Link */}
          <a
            href={docsUrl}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-md border border-slate-700/70 bg-slate-800/60 hover:bg-slate-800 text-xs font-medium text-slate-200 hover:text-white transition-colors focus:outline-none focus:ring-2 focus:ring-indigo-500"
          >
            <BookOpen className="w-3.5 h-3.5 text-slate-400" />
            <span>API Docs</span>
          </a>
        </div>
      </div>
    </header>
  );
};
