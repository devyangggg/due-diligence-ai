'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import type { AgentName, AgentState, AnalysisEvent, Run } from '@/types';
import { createInitialAgents } from '@/types';

const WS_URL = 'ws://localhost:8000/test';
const PARALLEL_AGENTS: AgentName[] = ['summary', 'risk', 'info'];
const STAGES: AgentName[] = ['fundamentals', ...PARALLEL_AGENTS, 'synthesis'];

export function useAnalysis() {
  const [agents, setAgents] = useState<Record<AgentName, AgentState>>(createInitialAgents);
  const [run, setRun] = useState<Run | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => () => { wsRef.current?.close(); }, []);

  const runAnalysis = useCallback((ticker: string): Promise<Run | null> => {
    if (wsRef.current) return Promise.resolve(null);
    setRun(null);
    setError(null);
    setIsRunning(true);
    const initial = createInitialAgents();
    initial.fundamentals.status = 'running';
    setAgents(initial);

    return new Promise((resolve) => {
      let ws: WebSocket;
      let settled = false;
      const finish = (result: Run | null, message?: string) => {
        if (settled) return;
        settled = true;
        setRun(result);
        setError(message ?? null);
        setIsRunning(false);
        if (wsRef.current === ws) wsRef.current = null;
        if (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING) ws.close();
        resolve(result);
      };

      try {
        ws = new WebSocket(WS_URL);
        wsRef.current = ws;
      } catch {
        setError('Failed to establish WebSocket connection.');
        setIsRunning(false);
        resolve(null);
        return;
      }

      ws.onopen = () => ws.send(JSON.stringify({ ticker }));
      ws.onmessage = (event) => {
        let message: AnalysisEvent;
        try { message = JSON.parse(event.data); } catch { return; }
        if (settled) return;
        if (message.type === 'stage' && STAGES.includes(message.stage) && message.status === 'completed') {
          setAgents((previous) => {
            const next = { ...previous, [message.stage]: { ...previous[message.stage], status: 'complete' as const } };
            if (message.stage === 'fundamentals') {
              for (const stage of PARALLEL_AGENTS) {
                if (next[stage].status === 'idle') next[stage] = { ...next[stage], status: 'running' };
              }
            }
            if (PARALLEL_AGENTS.every((stage) => next[stage].status === 'complete') && next.synthesis.status === 'idle') {
              next.synthesis = { ...next.synthesis, status: 'running' };
            }
            return next;
          });
        } else if (message.type === 'final' && message.run?.status === 'completed') {
          setAgents((previous) => Object.fromEntries(STAGES.map((stage) => [stage, { ...previous[stage], status: 'complete' }])) as Record<AgentName, AgentState>);
          finish(message.run);
        } else if (message.type === 'error') {
          setAgents((previous) => Object.fromEntries(STAGES.map((stage) => [stage, { ...previous[stage], status: previous[stage].status === 'running' ? 'error' : previous[stage].status }])) as Record<AgentName, AgentState>);
          finish(message.run ?? null, message.message || message.run?.error || 'Analysis failed.');
        }
      };
      ws.onerror = () => finish(null, 'Connection failed — is the backend running on localhost:8000?');
      ws.onclose = () => finish(null, 'Analysis disconnected before completion. The run may have been interrupted.');
    });
  }, []);

  const clearError = useCallback(() => setError(null), []);
  const loadResult = useCallback((result: Run) => {
    const states = createInitialAgents();
    if (result.status === 'completed') {
      for (const stage of STAGES) states[stage].status = 'complete';
    }
    setAgents(states);
    setRun(result);
    setError(result.status === 'completed' ? null : result.error || `Analysis ${result.status}.`);
    setIsRunning(false);
  }, []);
  const resetState = useCallback(() => {
    setAgents(createInitialAgents());
    setRun(null);
    setError(null);
    setIsRunning(false);
  }, []);

  return { agents, run, isRunning, error, runAnalysis, clearError, loadResult, resetState };
}
