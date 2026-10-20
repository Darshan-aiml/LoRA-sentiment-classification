import React, { FormEvent, KeyboardEvent } from 'react';
import { Send, Trash2, AlertCircle, Loader2 } from 'lucide-react';

interface SentimentInputProps {
  value: string;
  onChange: (value: string) => void;
  onSubmit: () => void;
  onClear: () => void;
  isLoading: boolean;
  error: string | null;
  maxLength?: number;
}

export const SentimentInput: React.FC<SentimentInputProps> = ({
  value,
  onChange,
  onSubmit,
  onClear,
  isLoading,
  error,
  maxLength = 1000,
}) => {
  const trimmed = value.trim();
  const isBlank = trimmed.length === 0;
  const isOverLimit = value.length > maxLength;
  const charsRemaining = maxLength - value.length;

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    if (!isBlank && !isOverLimit && !isLoading) {
      onSubmit();
    }
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
      e.preventDefault();
      if (!isBlank && !isOverLimit && !isLoading) {
        onSubmit();
      }
    }
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-3">
      <div className="relative rounded-xl border border-slate-800 bg-slate-900/60 focus-within:border-indigo-500/80 focus-within:ring-1 focus-within:ring-indigo-500/80 transition-all shadow-sm">
        <textarea
          value={value}
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Enter English text to analyze sentiment (e.g. review, opinion, critique)..."
          rows={5}
          disabled={isLoading}
          aria-label="Sentiment input text"
          className="w-full bg-transparent p-4 text-sm text-slate-100 placeholder-slate-500 resize-y min-h-[120px] focus:outline-none disabled:opacity-60 leading-relaxed font-sans"
        />

        {/* Bottom bar with counter and action buttons */}
        <div className="flex items-center justify-between px-4 py-2.5 border-t border-slate-800/80 bg-slate-950/40 text-xs">
          <div className="flex items-center gap-2">
            <span
              className={`font-mono text-xs ${
                isOverLimit
                  ? 'text-rose-400 font-semibold'
                  : charsRemaining < 100
                  ? 'text-amber-400'
                  : 'text-slate-500'
              }`}
            >
              {value.length} / {maxLength} chars
            </span>
            <span className="hidden sm:inline text-slate-600 text-[11px]">
              • ⌘+Enter to run
            </span>
          </div>

          <div className="flex items-center gap-2">
            {value.length > 0 && (
              <button
                type="button"
                onClick={onClear}
                disabled={isLoading}
                className="inline-flex items-center gap-1 px-2.5 py-1.5 rounded-md text-xs text-slate-400 hover:text-slate-200 hover:bg-slate-800 transition-colors disabled:opacity-50"
              >
                <Trash2 className="w-3.5 h-3.5" />
                <span className="hidden sm:inline">Clear</span>
              </button>
            )}

            <button
              type="submit"
              disabled={isBlank || isOverLimit || isLoading}
              className="inline-flex items-center gap-1.5 px-4 py-1.5 rounded-md bg-indigo-600 hover:bg-indigo-500 active:bg-indigo-700 text-white font-medium text-xs shadow-sm transition-all focus:outline-none focus:ring-2 focus:ring-indigo-500 disabled:opacity-50 disabled:pointer-events-none"
            >
              {isLoading ? (
                <>
                  <Loader2 className="w-3.5 h-3.5 animate-spin" />
                  <span>Analyzing...</span>
                </>
              ) : (
                <>
                  <Send className="w-3.5 h-3.5" />
                  <span>Analyze Sentiment</span>
                </>
              )}
            </button>
          </div>
        </div>
      </div>

      {/* Error Message Box */}
      {error && (
        <div
          role="alert"
          className="flex items-start gap-2.5 p-3 rounded-lg bg-rose-950/40 border border-rose-800/60 text-rose-300 text-xs animate-in fade-in duration-200"
        >
          <AlertCircle className="w-4 h-4 text-rose-400 shrink-0 mt-0.5" />
          <div className="flex-1">
            <span className="font-semibold block text-rose-200">Analysis Error</span>
            <span className="text-rose-300 leading-relaxed">{error}</span>
          </div>
        </div>
      )}
    </form>
  );
};
