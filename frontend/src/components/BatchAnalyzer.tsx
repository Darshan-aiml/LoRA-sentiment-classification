import React, { useState } from 'react';
import { Layers, Send, Trash2, AlertCircle, Loader2, FileText, CheckCircle2, XCircle } from 'lucide-react';
import { predictBatch } from '../services/api';
import { BatchPredictResponse } from '../types/api';

const MAX_BATCH_ITEMS = 32;

const SAMPLE_BATCH = [
  "This product exceeded all my expectations, wonderfully engineered!",
  "A total waste of money, stopped working after three days.",
  "Decent performance, but the setup process was confusing and frustrating.",
  "Truly captivating storyline with heartfelt, memorable performances.",
].join('\n');

export const BatchAnalyzer: React.FC = () => {
  const [inputText, setInputText] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [results, setResults] = useState<BatchPredictResponse | null>(null);

  // Extract non-empty lines
  const lines = inputText
    .split('\n')
    .map((l) => l.trim())
    .filter((l) => l.length > 0);

  const lineCount = lines.length;
  const isOverLimit = lineCount > MAX_BATCH_ITEMS;
  const isBlank = lineCount === 0;

  const handleRunBatch = async () => {
    if (isBlank || isOverLimit || isLoading) return;

    setIsLoading(true);
    setError(null);
    try {
      const response = await predictBatch(lines);
      setResults(response);
    } catch (err: unknown) {
      if (err && typeof err === 'object' && 'message' in err) {
        setError((err as { message: string }).message);
      } else {
        setError('Batch inference failed. Please check the backend connection.');
      }
    } finally {
      setIsLoading(false);
    }
  };

  const handleLoadSample = () => {
    setInputText(SAMPLE_BATCH);
    setResults(null);
    setError(null);
  };

  const handleClear = () => {
    setInputText('');
    setResults(null);
    setError(null);
  };

  return (
    <div className="space-y-6">
      {/* Batch Input Form */}
      <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-5 space-y-3.5 shadow-sm">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 pb-2 border-b border-slate-800">
          <div>
            <h3 className="text-sm font-semibold text-slate-200 flex items-center gap-1.5">
              <Layers className="w-4 h-4 text-indigo-400" />
              <span>Batch Sentiment Evaluation</span>
            </h3>
            <p className="text-xs text-slate-400">
              Evaluate up to 32 independent texts simultaneously (one sentence per line).
            </p>
          </div>
          <button
            type="button"
            onClick={handleLoadSample}
            disabled={isLoading}
            className="text-xs text-indigo-400 hover:text-indigo-300 font-medium inline-flex items-center gap-1 self-start sm:self-auto"
          >
            <FileText className="w-3.5 h-3.5" />
            <span>Load Sample Batch</span>
          </button>
        </div>

        <div className="relative rounded-lg border border-slate-800 bg-slate-950/60 focus-within:border-indigo-500/80 transition-colors">
          <textarea
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            placeholder={`Enter texts to analyze, separated by newlines...\nLine 1: Excellent quality and very fast shipping.\nLine 2: Disappointing material, broke immediately.`}
            rows={6}
            disabled={isLoading}
            aria-label="Batch sentiment input lines"
            className="w-full bg-transparent p-3.5 text-xs font-mono text-slate-200 placeholder-slate-500 resize-y focus:outline-none disabled:opacity-60 leading-relaxed"
          />

          <div className="flex items-center justify-between px-3.5 py-2 border-t border-slate-800/80 bg-slate-950/80 text-xs">
            <span
              className={`font-mono text-[11px] ${
                isOverLimit
                  ? 'text-rose-400 font-semibold'
                  : lineCount > 0
                  ? 'text-indigo-300'
                  : 'text-slate-500'
              }`}
            >
              {lineCount} / {MAX_BATCH_ITEMS} items detected
            </span>

            <div className="flex items-center gap-2">
              {inputText.length > 0 && (
                <button
                  type="button"
                  onClick={handleClear}
                  disabled={isLoading}
                  className="px-2.5 py-1 text-[11px] text-slate-400 hover:text-slate-200 inline-flex items-center gap-1"
                >
                  <Trash2 className="w-3 h-3" />
                  <span>Clear</span>
                </button>
              )}

              <button
                type="button"
                onClick={handleRunBatch}
                disabled={isBlank || isOverLimit || isLoading}
                className="inline-flex items-center gap-1.5 px-3 py-1 rounded-md bg-indigo-600 hover:bg-indigo-500 active:bg-indigo-700 text-white font-medium text-xs shadow-sm transition-all focus:outline-none focus:ring-2 focus:ring-indigo-500 disabled:opacity-50 disabled:pointer-events-none"
              >
                {isLoading ? (
                  <>
                    <Loader2 className="w-3.5 h-3.5 animate-spin" />
                    <span>Evaluating Batch...</span>
                  </>
                ) : (
                  <>
                    <Send className="w-3.5 h-3.5" />
                    <span>Run Batch ({lineCount})</span>
                  </>
                )}
              </button>
            </div>
          </div>
        </div>

        {error && (
          <div
            role="alert"
            className="flex items-start gap-2.5 p-3 rounded-lg bg-rose-950/40 border border-rose-800/60 text-rose-300 text-xs"
          >
            <AlertCircle className="w-4 h-4 text-rose-400 shrink-0 mt-0.5" />
            <div>
              <span className="font-semibold block text-rose-200">Batch Error</span>
              <span>{error}</span>
            </div>
          </div>
        )}
      </div>

      {/* Batch Results Table */}
      {results && (
        <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-5 space-y-4 shadow-sm animate-in fade-in duration-300">
          <div className="flex items-center justify-between pb-3 border-b border-slate-800">
            <div>
              <h4 className="text-sm font-semibold text-slate-200">
                Batch Evaluation Results ({results.count} items)
              </h4>
              <p className="text-xs text-slate-400">
                Predictions returned in identical order with per-class softmax probabilities.
              </p>
            </div>
            <span className="text-xs font-mono px-2.5 py-1 rounded bg-slate-800 text-slate-300 border border-slate-700">
              Rank {results.predictions[0]?.rank ?? 16}
            </span>
          </div>

          <div className="space-y-2.5">
            {results.predictions.map((item, index) => {
              const isPos = item.sentiment === 'positive';
              const posPercent = (item.probabilities.positive * 100).toFixed(1);
              const negPercent = (item.probabilities.negative * 100).toFixed(1);

              return (
                <div
                  key={index}
                  className="p-3.5 rounded-lg border border-slate-800/80 bg-slate-950/60 flex flex-col sm:flex-row sm:items-center justify-between gap-3 text-xs"
                >
                  <div className="flex items-start gap-2.5 flex-1 min-w-0">
                    <span className="font-mono text-slate-500 font-semibold w-5 text-right shrink-0 mt-0.5">
                      #{index + 1}
                    </span>
                    <div className="min-w-0 flex-1">
                      <p className="text-slate-200 font-sans leading-relaxed break-words">
                        "{item.text}"
                      </p>
                      <div className="flex items-center gap-3 mt-1.5 text-[11px] font-mono text-slate-400">
                        <span>Pos: {posPercent}%</span>
                        <span>•</span>
                        <span>Neg: {negPercent}%</span>
                      </div>
                    </div>
                  </div>

                  {/* Sentiment and Confidence Tag */}
                  <div className="flex sm:flex-col items-center sm:items-end justify-between sm:justify-center gap-1 shrink-0 pt-2 sm:pt-0 border-t sm:border-t-0 border-slate-800/60">
                    <span
                      className={`inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-xs font-semibold capitalize ${
                        isPos
                          ? 'bg-emerald-950/80 text-emerald-300 border border-emerald-800/60'
                          : 'bg-rose-950/80 text-rose-300 border border-rose-800/60'
                      }`}
                    >
                      {isPos ? (
                        <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
                      ) : (
                        <XCircle className="w-3.5 h-3.5 text-rose-400" />
                      )}
                      <span>{item.sentiment}</span>
                    </span>
                    <span className="font-mono text-[11px] text-slate-400">
                      {(item.confidence * 100).toFixed(2)}% conf
                    </span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
};
