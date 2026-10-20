import React from 'react';
import { MessageSquareText } from 'lucide-react';

interface ExamplePromptsProps {
  onSelectPrompt: (prompt: string) => void;
  disabled?: boolean;
}

interface ExampleItem {
  label: string;
  category: 'positive' | 'negative' | 'nuanced';
  text: string;
}

const EXAMPLES: ExampleItem[] = [
  {
    label: 'Strong Positive',
    category: 'positive',
    text: 'This product is fantastic and works perfectly right out of the box.',
  },
  {
    label: 'Subtle Positive',
    category: 'positive',
    text: 'A quiet, thoughtful indie film with remarkably genuine performances.',
  },
  {
    label: 'Strong Negative',
    category: 'negative',
    text: 'Extremely disappointing and frustrating experience, completely broken.',
  },
  {
    label: 'Nuanced Negative',
    category: 'negative',
    text: 'Despite stunning visual effects and talented leads, the clumsy plot fell flat.',
  },
];

export const ExamplePrompts: React.FC<ExamplePromptsProps> = ({
  onSelectPrompt,
  disabled = false,
}) => {
  return (
    <div className="space-y-2">
      <div className="flex items-center gap-1.5 text-xs font-mono text-slate-400 uppercase tracking-wider">
        <MessageSquareText className="w-3.5 h-3.5 text-indigo-400" />
        <span>Try Example Prompts</span>
      </div>
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
        {EXAMPLES.map((example, idx) => (
          <button
            key={idx}
            type="button"
            disabled={disabled}
            onClick={() => onSelectPrompt(example.text)}
            className="text-left p-2.5 rounded-lg border border-slate-800 bg-slate-900/40 hover:bg-slate-800/80 hover:border-slate-700 text-xs text-slate-300 transition-all focus:outline-none focus:ring-1 focus:ring-indigo-500 disabled:opacity-50 disabled:pointer-events-none group"
          >
            <div className="flex items-center justify-between mb-1">
              <span className="font-semibold text-[11px] text-slate-200 group-hover:text-white">
                {example.label}
              </span>
              <span
                className={`text-[9px] font-mono px-1.5 py-0.2 rounded uppercase ${
                  example.category === 'positive'
                    ? 'bg-emerald-950/70 text-emerald-400 border border-emerald-800/40'
                    : 'bg-rose-950/70 text-rose-400 border border-rose-800/40'
                }`}
              >
                {example.category}
              </span>
            </div>
            <p className="text-slate-400 group-hover:text-slate-300 line-clamp-2 text-[11px] leading-relaxed">
              "{example.text}"
            </p>
          </button>
        ))}
      </div>
    </div>
  );
};
