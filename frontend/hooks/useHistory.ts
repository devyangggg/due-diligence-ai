'use client';

import { useState, useEffect, useCallback } from 'react';
import type { Run } from '@/types';

const RUNS_URL = 'http://localhost:8000/runs';

export function useHistory() {
  const [entries, setEntries] = useState<Run[]>([]);
  const [error, setError] = useState<string | null>(null);

  const reload = useCallback(async () => {
    try {
      const response = await fetch(RUNS_URL, { cache: 'no-store' });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const runs: Run[] = await response.json();
      setEntries(runs);
      setError(null);
    } catch {
      setError('Could not load analysis history from the server.');
    }
  }, []);

  useEffect(() => { void reload(); }, [reload]);

  const getRun = useCallback(async (id: string): Promise<Run | null> => {
    try {
      const response = await fetch(`${RUNS_URL}/${encodeURIComponent(id)}`, { cache: 'no-store' });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const run: Run = await response.json();
      setError(null);
      return run;
    } catch {
      setError('Could not load this analysis from the server.');
      return null;
    }
  }, []);

  return { entries, error, reload, getRun };
}
