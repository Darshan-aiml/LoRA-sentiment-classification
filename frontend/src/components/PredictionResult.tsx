import React from 'react';
import { CheckCircle2, XCircle, Info, Sparkles, Layers } from 'lucide-react';
import { PredictResponse } from '../types/api';
import { ProbabilityBars } from './ProbabilityBars';

interface PredictionResultProps {
  result: PredictResponse | null;
}

export const PredictionResult: React.FC<PredictionResultProps> = ({ result }) => {
  if (!result) {
    return (
      <div className="h-full min-h-[300px] flex flex-col items-center justify-center p-8 rounded-xl border border-dashed border-slate-800 bg-slate-900/30 text-center">
        <div className="w-12 h-12 rounded-full bg-slate-800/60 flex items-center justify-center text-slate-500 mb-3">
          <Sparkles className="w-5 h-5 text-slate-400" />
        </div>
        <h3 className="text-sm font-medium text-slate-300 mb-1">No Prediction Yet</h3>
        <p className="text-xs text-slate-500 max-w-xs">
          Enter text or select an example prompt on the left to evaluate sentiment with the fine-tuned LoRA model.
        </p>
      </div>
    );
  }

  const isPositive = result.sentiment === 'positive';
  const confidencePercent = (result.confidence * 100).toFixed(2);

  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-5 sm:p-6 space-y-5 shadow-sm">
      {/* Top Banner: Predicted Class & Confidence */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-slate-800">
        <div className="flex items-center gap-3">
          <div
            className={`w-12 h-12 rounded-xl flex items-center justify-center ${
              isPositive
                ? 'bg-emerald-950/80 border border-emerald-700/50 text-emerald-400'
                : 'bg-rose-950/80 border border-rose-700/50 text-rose-400'
            }`}
          >
            {isPositive ? (
              <CheckCircle2 className="w-6 h-6 text-emerald-400" />
            ) : (
              <XCircle className="w-6 h-6 text-rose-400" />
            )}
          </div>
          <div>
            <span className="text-xs font-mono uppercase tracking-wider text-slate-400">
              Predicted Sentiment
            </span>
            <div className="flex items-center gap-2">
              <span
                className={`text-xl sm:text-2xl font-bold tracking-tight capitalize ${
                  isPositive ? 'text-emerald-400' : 'text-rose-400'
                }`}
              >
                {result.sentiment}
              </span>
              <span
                className={`text-xs px-2 py-0.5 rounded-full font-mono font-medium ${
                  isPositive
                    ? 'bg-emerald-950 text-emerald-300 border border-emerald-800/60'
                    : 'bg-rose-950 text-rose-300 border border-rose-800/60'
                }`}
              >
                Class {isPositive ? '1' : '0'}
              </span>
            </div>
          </div>
        </div>

        {/* Confidence metric */}
        <div className="text-left sm:text-right">
          <span className="text-xs font-mono uppercase tracking-wider text-slate-400">
            Softmax Confidence
          </span>
          <div className="text-2xl font-mono font-semibold text-slate-100">
            {confidencePercent}%
          </div>
        </div>
      </div>

      {/* Probability Distribution Bars */}
      <div>
        <h4 className="text-xs font-mono uppercase tracking-wider text-slate-400 mb-3 flex items-center gap-1.5">
          <Layers className="w-3.5 h-3.5 text-slate-400" />
          <span>Softmax Probability Distribution</span>
        </h4>
        <ProbabilityBars
          probabilities={result.probabilities}
          predictedSentiment={result.sentiment}
        />
      </div>

      {/* Model & Fine-Tuning Metadata */}
      <div className="grid grid-cols-2 sm:grid-cols-3 gap-2.5 pt-3 border-t border-slate-800 text-xs">
        <div className="p-2.5 rounded-lg bg-slate-950/60 border border-slate-800/80">
          <span className="text-slate-500 font-mono block text-[10px] uppercase">
            Base Architecture
          </span>
          <span className="text-slate-200 font-mono font-medium truncate block" title={result.model}>
            {result.model}
          </span>
        </div>
        <div className="p-2.5 rounded-lg bg-slate-950/60 border border-slate-800/80">
          <span className="text-slate-500 font-mono block text-[10px] uppercase">
            Fine-Tuning Method
          </span>
          <span className="text-indigo-300 font-medium font-mono">
            PEFT / {result.adapter}
          </span>
        </div>
        <div className="p-2.5 rounded-lg bg-slate-950/60 border border-slate-800/80 col-span-2 sm:col-span-1">
          <span className="text-slate-500 font-mono block text-[10px] uppercase">
            Adapter Rank (r)
          </span>
          <span className="text-slate-200 font-mono font-medium">
            r = {result.rank} (alpha = 32)
          </span>
        </div>
      </div>

      {/* Calibration and Binary Scope Notice */}
      <div className="flex items-start gap-2.5 p-3 rounded-lg bg-slate-950/70 border border-slate-800/60 text-slate-400 text-xs leading-relaxed">
        <Info className="w-4 h-4 text-slate-400 shrink-0 mt-0.5" />
        <p>
          <strong className="text-slate-300">Calibration & Scope Note:</strong> Softmax output scores represent uncalibrated relative class weights rather than Bayesian confidence. The model evaluates binary SST-2 sentiment (0=negative, 1=positive) and does not predict an explicit neutral class.
        </p>
      </div>
    </div>
  );
};
