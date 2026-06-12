'use client';

import { useState, useEffect, useCallback } from 'react';
import type { HistoryEntry, AnalysisResult } from '@/types';

const STORAGE_KEY = 'dd-history';

export function useHistory() {
  const [entries, setEntries] = useState<HistoryEntry[]>([]);

  // Load from localStorage on mount
  useEffect(() => {
    if (typeof window === 'undefined') return;
    try {
      const stored = localStorage.getItem(STORAGE_KEY);
      if (stored) {
        setEntries(JSON.parse(stored));
      }
    } catch {
      // Corrupted data — start fresh
      localStorage.removeItem(STORAGE_KEY);
    }
  }, []);

  // Persist to localStorage whenever entries change
  const persist = useCallback((updated: HistoryEntry[]) => {
    setEntries(updated);
    if (typeof window !== 'undefined') {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(updated));
    }
  }, []);

  const addEntry = useCallback(
    (result: AnalysisResult) => {
      const entry: HistoryEntry = {
        id: `${result.ticker}-${Date.now()}`,
        ticker: result.ticker,
        timestamp: result.timestamp,
        result,
      };
      // Prepend new entry, keep max 20
      persist([entry, ...entries].slice(0, 20));
    },
    [entries, persist]
  );

  const clearHistory = useCallback(() => {
    persist([]);
  }, [persist]);

  return { entries, addEntry, clearHistory };
}
