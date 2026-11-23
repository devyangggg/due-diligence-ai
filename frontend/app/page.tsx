'use client';

import { useState } from 'react';
import Sidebar from '@/components/Sidebar';
import TickerInput from '@/components/TickerInput';
import AgentStatusPanel from '@/components/AgentStatus';
import BriefView from '@/components/BriefView';
import ErrorBanner from '@/components/ErrorBanner';
import { useAnalysis } from '@/hooks/useAnalysis';
import { useHistory } from '@/hooks/useHistory';
import type { Run } from '@/types';

export default function Home() {
  const { agents, run, isRunning, error, runAnalysis, clearError, loadResult, resetState } = useAnalysis();
  const { entries, error: historyError, reload, getRun } = useHistory();
  const [activeHistoryId, setActiveHistoryId] = useState<string | null>(null);
  const [currentTicker, setCurrentTicker] = useState<string | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(false);

  const handleRunAnalysis = async (ticker: string) => {
    setActiveHistoryId(null);
    setCurrentTicker(ticker);
    const result = await runAnalysis(ticker);
    if (result) setActiveHistoryId(result.id);
    await reload();
  };

  const handleSelectHistory = async (entry: Run) => {
    if (isRunning) return;
    const selected = await getRun(entry.id);
    if (!selected) return;
    setActiveHistoryId(selected.id);
    setCurrentTicker(selected.ticker);
    loadResult(selected);
  };

  const handleNewAnalysis = () => {
    if (isRunning) return;
    setActiveHistoryId(null);
    setCurrentTicker(null);
    resetState();
  };

  const showWelcome = !isRunning && !currentTicker && !error;

  return (
    <div className="app-layout">
      <Sidebar
        entries={entries}
        activeId={activeHistoryId}
        isOpen={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
        onSelect={(entry) => { void handleSelectHistory(entry); }}
        onNewAnalysis={handleNewAnalysis}
      />
      <main className="main-panel">
        <div className="main-content">
          <div className="mobile-topbar">
            <button className="hamburger-btn" onClick={() => setSidebarOpen(true)} aria-label="Open menu">
              <svg width="22" height="22" viewBox="0 0 22 22" fill="none">
                <path d="M4 6h14M4 11h14M4 16h14" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
              </svg>
            </button>
            <span className="mobile-topbar-title">Due Diligence AI</span>
          </div>
          {error && <ErrorBanner message={error} onDismiss={clearError} />}
          {historyError && <ErrorBanner message={historyError} onDismiss={() => { void reload(); }} />}
          <TickerInput isRunning={isRunning} onSubmit={handleRunAnalysis} />
          {showWelcome ? (
            <div className="welcome">
              <svg className="welcome-icon" viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">
                <rect x="6" y="4" width="36" height="40" rx="3" stroke="currentColor" strokeWidth="2" />
                <path d="M14 14h20M14 22h20M14 30h12" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
              </svg>
              <h2 className="welcome-title">Financial Due Diligence</h2>
              <p className="welcome-text">
                Enter a stock ticker above to run a multi-agent analysis.
                Four specialized agents will analyze fundamentals, SEC filings,
                risk metrics, and insider activity — then synthesize a
                comprehensive due diligence brief.
              </p>
            </div>
          ) : (
            <>
              {currentTicker && <AgentStatusPanel agents={agents} />}
              <BriefView run={run} isRunning={isRunning} />
            </>
          )}
        </div>
      </main>
    </div>
  );
}
