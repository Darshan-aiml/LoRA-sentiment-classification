import { useState, useEffect, useCallback } from 'react';
import { Header } from './components/Header';
import { SentimentInput } from './components/SentimentInput';
import { PredictionResult } from './components/PredictionResult';
import { ExamplePrompts } from './components/ExamplePrompts';
import { BatchAnalyzer } from './components/BatchAnalyzer';
import { ApiDiagnostics } from './components/ApiDiagnostics';
import { fetchHealth, predictSentiment } from './services/api';
import { HealthResponse, PredictResponse } from './types/api';
import { MessageSquare, Layers, ShieldCheck } from 'lucide-react';

export function App() {
  const [inputText, setInputText] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<PredictResponse | null>(null);

  const [activeTab, setActiveTab] = useState<'single' | 'batch'>('single');

  // Backend Health
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [loadingHealth, setLoadingHealth] = useState(true);

  const checkHealth = useCallback(async () => {
    setLoadingHealth(true);
    try {
      const data = await fetchHealth();
      setHealth(data);
    } catch {
      setHealth({
        status: 'unavailable',
        model_loaded: false,
        device: 'unknown',
        base_model: 'distilbert/distilbert-base-uncased',
        adapter_path: 'Models/lora-sst2-adapter',
        details: 'Failed to connect to backend inference server.',
      });
    } finally {
      setLoadingHealth(false);
    }
  }, []);

  useEffect(() => {
    checkHealth();
  }, [checkHealth]);

  const handlePredict = async () => {
    if (!inputText.trim() || isLoading) return;

    setIsLoading(true);
    setError(null);
    try {
      const response = await predictSentiment(inputText);
      setResult(response);
    } catch (err: unknown) {
      if (err && typeof err === 'object' && 'message' in err) {
        setError((err as { message: string }).message);
      } else {
        setError('Inference failed. Please check the backend connection.');
      }
    } finally {
      setIsLoading(false);
    }
  };

  const handleSelectPrompt = (prompt: string) => {
    setInputText(prompt);
    setError(null);
  };

  const handleClear = () => {
    setInputText('');
    setResult(null);
    setError(null);
  };

  return (
    <div className="min-h-screen flex flex-col bg-[#090d16] text-slate-100">
      <Header
        health={health}
        loadingHealth={loadingHealth}
        onRefreshHealth={checkHealth}
      />

      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
        {/* Navigation Tabs */}
        <div className="flex items-center justify-between border-b border-slate-800 pb-3 flex-wrap gap-4">
          <div className="flex items-center gap-2">
            <button
              onClick={() => setActiveTab('single')}
              className={`inline-flex items-center gap-2 px-3.5 py-2 rounded-lg text-xs font-medium transition-all ${
                activeTab === 'single'
                  ? 'bg-indigo-600/20 text-indigo-300 border border-indigo-500/40 font-semibold'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900 border border-transparent'
              }`}
            >
              <MessageSquare className="w-3.5 h-3.5" />
              <span>Single Text Analyzer</span>
            </button>

            <button
              onClick={() => setActiveTab('batch')}
              className={`inline-flex items-center gap-2 px-3.5 py-2 rounded-lg text-xs font-medium transition-all ${
                activeTab === 'batch'
                  ? 'bg-indigo-600/20 text-indigo-300 border border-indigo-500/40 font-semibold'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900 border border-transparent'
              }`}
            >
              <Layers className="w-3.5 h-3.5" />
              <span>Batch Evaluation (up to 32)</span>
            </button>
          </div>

          <div className="hidden sm:flex items-center gap-2 text-xs font-mono text-slate-400">
            <ShieldCheck className="w-4 h-4 text-emerald-400" />
            <span>SST-2 Benchmark Validated (89.4% Heldout Acc)</span>
          </div>
        </div>

        {/* Tab 1: Single Prediction */}
        {activeTab === 'single' ? (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
            {/* Left Column: Input and Example Prompts */}
            <div className="lg:col-span-7 space-y-5">
              <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-5 space-y-3.5 shadow-sm">
                <div className="flex items-center justify-between">
                  <h2 className="text-sm font-semibold text-slate-200">
                    Input Text
                  </h2>
                  <span className="text-[11px] font-mono text-slate-400">
                    Max 1,000 characters
                  </span>
                </div>

                <SentimentInput
                  value={inputText}
                  onChange={setInputText}
                  onSubmit={handlePredict}
                  onClear={handleClear}
                  isLoading={isLoading}
                  error={error}
                  maxLength={1000}
                />
              </div>

              <div className="rounded-xl border border-slate-800 bg-slate-900/40 p-4">
                <ExamplePrompts
                  onSelectPrompt={handleSelectPrompt}
                  disabled={isLoading}
                />
              </div>
            </div>

            {/* Right Column: Prediction Results Card */}
            <div className="lg:col-span-5">
              <PredictionResult result={result} />
            </div>
          </div>
        ) : (
          /* Tab 2: Batch Prediction */
          <BatchAnalyzer />
        )}

        {/* API Diagnostics Section */}
        <section aria-labelledby="diagnostics-heading">
          <h2 id="diagnostics-heading" className="sr-only">
            API Health & System Diagnostics
          </h2>
          <ApiDiagnostics
            health={health}
            loading={loadingHealth}
            onRefresh={checkHealth}
          />
        </section>
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-800/80 bg-slate-950/70 py-6 text-center text-xs text-slate-400">
        <div className="max-w-7xl mx-auto px-4 flex flex-col sm:flex-row items-center justify-between gap-3">
          <p>
            LoRA Sentiment AI — Parameter-Efficient Fine-Tuning Study with DistilBERT on SST-2.
          </p>
          <div className="flex items-center gap-4 text-[11px] font-mono">
            <span>FastAPI Backend</span>
            <span>•</span>
            <span>PyTorch & PEFT</span>
            <span>•</span>
            <span>React + Vite</span>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;
