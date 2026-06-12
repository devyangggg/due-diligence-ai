'use client';

import { useState, type FormEvent } from 'react';

interface TickerInputProps {
  isRunning: boolean;
  onSubmit: (ticker: string) => void;
}

export default function TickerInput({ isRunning, onSubmit }: TickerInputProps) {
  const [ticker, setTicker] = useState('');

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    const trimmed = ticker.trim().toUpperCase();
    if (!trimmed || isRunning) return;
    onSubmit(trimmed);
  };

  return (
    <form onSubmit={handleSubmit} className="ticker-form">
      <div className="ticker-input-group">
        <label htmlFor="ticker-input" className="ticker-label">
          Stock Ticker
        </label>
        <div className="ticker-row">
          <input
            id="ticker-input"
            type="text"
            value={ticker}
            onChange={(e) => setTicker(e.target.value.toUpperCase())}
            placeholder="e.g. AAPL, MSFT, GOOGL"
            className="ticker-input"
            disabled={isRunning}
            autoComplete="off"
            spellCheck={false}
          />
          <button
            id="run-analysis-btn"
            type="submit"
            className="run-btn"
            disabled={isRunning || !ticker.trim()}
          >
            {isRunning ? (
              <>
                <span className="spinner" />
                Analyzing…
              </>
            ) : (
              'Run Analysis'
            )}
          </button>
        </div>
      </div>
    </form>
  );
}
