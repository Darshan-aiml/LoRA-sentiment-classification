import React from 'react';
import { SentimentProbabilities } from '../types/api';

interface ProbabilityBarsProps {
  probabilities: SentimentProbabilities;
  predictedSentiment: 'positive' | 'negative';
}

export const ProbabilityBars: React.FC<ProbabilityBarsProps> = ({
  probabilities,
  predictedSentiment,
}) => {
  const posPercent = Math.max(0, Math.min(100, probabilities.positive * 100));
  const negPercent = Math.max(0, Math.min(100, probabilities.negative * 100));

  const isPositiveWinner = predictedSentiment === 'positive';

  return (
    <div className="space-y-3.5">
      {/* Positive Class Bar */}
      <div>
        <div className="flex justify-between items-center text-xs mb-1.5">
          <div className="flex items-center gap-1.5">
            <span
              className={`w-2 h-2 rounded-full ${
                isPositiveWinner ? 'bg-emerald-400' : 'bg-slate-500'
              }`}
            />
            <span
              className={`font-medium ${
                isPositiveWinner ? 'text-emerald-300 font-semibold' : 'text-slate-400'
              }`}
            >
              Positive Class (1)
            </span>
            {isPositiveWinner && (
              <span className="text-[10px] font-mono px-1.5 py-0.2 rounded bg-emerald-950/80 border border-emerald-800/60 text-emerald-300">
                Predicted
              </span>
            )}
          </div>
          <span className="font-mono text-xs font-semibold text-slate-200">
            {posPercent.toFixed(2)}%
          </span>
        </div>
        <div
          role="progressbar"
          aria-label="Positive probability"
          aria-valuenow={posPercent}
          aria-valuemin={0}
          aria-valuemax={100}
          className="w-full h-2.5 bg-slate-900/90 rounded-full overflow-hidden border border-slate-800"
        >
          <div
            className={`h-full transition-all duration-500 rounded-full ${
              isPositiveWinner
                ? 'bg-gradient-to-r from-emerald-600 to-emerald-400'
                : 'bg-slate-600'
            }`}
            style={{ width: `${posPercent}%` }}
          />
        </div>
      </div>

      {/* Negative Class Bar */}
      <div>
        <div className="flex justify-between items-center text-xs mb-1.5">
          <div className="flex items-center gap-1.5">
            <span
              className={`w-2 h-2 rounded-full ${
                !isPositiveWinner ? 'bg-rose-400' : 'bg-slate-500'
              }`}
            />
            <span
              className={`font-medium ${
                !isPositiveWinner ? 'text-rose-300 font-semibold' : 'text-slate-400'
              }`}
            >
              Negative Class (0)
            </span>
            {!isPositiveWinner && (
              <span className="text-[10px] font-mono px-1.5 py-0.2 rounded bg-rose-950/80 border border-rose-800/60 text-rose-300">
                Predicted
              </span>
            )}
          </div>
          <span className="font-mono text-xs font-semibold text-slate-200">
            {negPercent.toFixed(2)}%
          </span>
        </div>
        <div
          role="progressbar"
          aria-label="Negative probability"
          aria-valuenow={negPercent}
          aria-valuemin={0}
          aria-valuemax={100}
          className="w-full h-2.5 bg-slate-900/90 rounded-full overflow-hidden border border-slate-800"
        >
          <div
            className={`h-full transition-all duration-500 rounded-full ${
              !isPositiveWinner
                ? 'bg-gradient-to-r from-rose-600 to-rose-400'
                : 'bg-slate-600'
            }`}
            style={{ width: `${negPercent}%` }}
          />
        </div>
      </div>
    </div>
  );
};
