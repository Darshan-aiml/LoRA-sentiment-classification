import React from 'react';
import { Activity, Server, Cpu, Database, RefreshCw, CheckCircle, AlertTriangle } from 'lucide-react';
import { HealthResponse } from '../types/api';
import { getApiBaseUrl } from '../services/api';

interface ApiDiagnosticsProps {
  health: HealthResponse | null;
  loading: boolean;
  onRefresh: () => void;
}

export const ApiDiagnostics: React.FC<ApiDiagnosticsProps> = ({
  health,
  loading,
  onRefresh,
}) => {
  const baseUrl = getApiBaseUrl();

  // Clean adapter path: sanitize full paths so system username/paths are never leaked in UI
  const cleanAdapterName = health?.adapter_path
    ? health.adapter_path.split(/[\\/]/).filter(Boolean).pop() || 'lora-sst2-adapter'
    : 'Unknown';

  const isReady = health?.status === 'ready' && health.model_loaded;

  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-4 sm:p-5 space-y-3.5 shadow-sm">
      <div className="flex items-center justify-between pb-2.5 border-b border-slate-800">
        <div className="flex items-center gap-2">
          <Activity className="w-4 h-4 text-indigo-400" />
          <h3 className="text-xs font-mono uppercase tracking-wider text-slate-300 font-semibold">
            API Health & System Diagnostics
          </h3>
        </div>
        <button
          onClick={onRefresh}
          disabled={loading}
          className="text-xs text-slate-400 hover:text-slate-200 inline-flex items-center gap-1 transition-colors disabled:opacity-50"
          title="Query GET /health"
        >
          <RefreshCw className={`w-3 h-3 ${loading ? 'animate-spin' : ''}`} />
          <span>Refresh</span>
        </button>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
        {/* Connection Status */}
        <div className="p-3 rounded-lg bg-slate-950/70 border border-slate-800/80">
          <div className="flex items-center gap-1.5 text-slate-500 font-mono text-[10px] uppercase mb-1">
            <Server className="w-3 h-3 text-slate-400" />
            <span>Connection</span>
          </div>
          <div className="flex items-center gap-1.5 font-medium">
            {isReady ? (
              <>
                <CheckCircle className="w-3.5 h-3.5 text-emerald-400" />
                <span className="text-emerald-300">Connected</span>
              </>
            ) : (
              <>
                <AlertTriangle className="w-3.5 h-3.5 text-rose-400" />
                <span className="text-rose-300">
                  {health?.status === 'initializing' ? 'Initializing' : 'Unavailable'}
                </span>
              </>
            )}
          </div>
        </div>

        {/* Compute Device */}
        <div className="p-3 rounded-lg bg-slate-950/70 border border-slate-800/80">
          <div className="flex items-center gap-1.5 text-slate-500 font-mono text-[10px] uppercase mb-1">
            <Cpu className="w-3 h-3 text-slate-400" />
            <span>Compute Device</span>
          </div>
          <span className="font-mono text-slate-200 font-medium">
            {health?.device ? health.device.toUpperCase() : 'N/A'}
          </span>
        </div>

        {/* Base Model */}
        <div className="p-3 rounded-lg bg-slate-950/70 border border-slate-800/80">
          <div className="flex items-center gap-1.5 text-slate-500 font-mono text-[10px] uppercase mb-1">
            <Database className="w-3 h-3 text-slate-400" />
            <span>Base Model</span>
          </div>
          <span
            className="font-mono text-slate-200 truncate block font-medium"
            title={health?.base_model || 'N/A'}
          >
            {health?.base_model || 'N/A'}
          </span>
        </div>

        {/* LoRA Adapter */}
        <div className="p-3 rounded-lg bg-slate-950/70 border border-slate-800/80">
          <div className="flex items-center gap-1.5 text-slate-500 font-mono text-[10px] uppercase mb-1">
            <Server className="w-3 h-3 text-slate-400" />
            <span>PEFT Adapter</span>
          </div>
          <span className="font-mono text-indigo-300 font-medium truncate block">
            {cleanAdapterName} (r={health?.lora_rank ?? 16})
          </span>
        </div>
      </div>

      {/* Diagnostic status banner if offline */}
      {!isReady && health?.details && (
        <div className="p-3 rounded-lg bg-rose-950/40 border border-rose-800/60 text-xs text-rose-300 space-y-1">
          <div className="flex items-center gap-1.5 font-semibold text-rose-200">
            <AlertTriangle className="w-4 h-4 text-rose-400 shrink-0" />
            <span>Connection Diagnostic</span>
          </div>
          <p className="font-mono text-[11px] leading-relaxed break-words">{health.details}</p>
        </div>
      )}

      {/* Backend URL information */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between text-[11px] font-mono text-slate-500 pt-2 border-t border-slate-800/60 gap-1">
        <span>Active Endpoint: <span className="text-slate-300 font-medium">{baseUrl}</span></span>
        {isReady && health?.details && (
          <span className="text-slate-400 italic truncate max-w-md" title={health.details}>
            Status: {health.details}
          </span>
        )}
      </div>
    </div>
  );
};
