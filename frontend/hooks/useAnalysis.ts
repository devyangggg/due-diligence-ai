'use client';

import { useState, useCallback, useRef } from 'react';
import type { AgentName, AgentState, AnalysisResult } from '@/types';
import { createInitialAgents } from '@/types';

const WS_URL = 'ws://localhost:8000/test';

// Map WebSocket message keys to our agent names
const MESSAGE_KEY_MAP: Record<string, AgentName> = {
  fundamentals_agent: 'fundamentals',
  summary_agent: 'summary',
  risk_agent: 'risk',
  info_agent: 'info',
  synth_agent: 'synthesis',
};

// Which agents depend on fundamentals completing first
const PARALLEL_AGENTS: AgentName[] = ['summary', 'risk', 'info'];

export function useAnalysis() {
  const [agents, setAgents] = useState<Record<AgentName, AgentState>>(createInitialAgents());
  const [finalBrief, setFinalBrief] = useState<string | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const completedRef = useRef<Set<AgentName>>(new Set());

  const runAnalysis = useCallback((ticker: string): Promise<AnalysisResult | null> => {
    return new Promise((resolve) => {
      // Reset state
      const freshAgents = createInitialAgents();
      setFinalBrief(null);
      setError(null);
      setIsRunning(true);
      completedRef.current = new Set();

      // Mark fundamentals as running (first agent to execute)
      freshAgents.fundamentals.status = 'running';
      setAgents({ ...freshAgents });

      let currentAgents = { ...freshAgents };
      let brief: string | null = null;

      try {
        const ws = new WebSocket(WS_URL);
        wsRef.current = ws;

        ws.onopen = () => {
          ws.send(ticker);
        };

        ws.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            const messageKey = Object.keys(data)[0];
            const agentName = MESSAGE_KEY_MAP[messageKey];

            if (!agentName) return;

            completedRef.current.add(agentName);

            // Update the completed agent
            currentAgents = {
              ...currentAgents,
              [agentName]: {
                ...currentAgents[agentName],
                status: 'complete' as const,
                data: data[messageKey],
              },
            };

            // Infer running states based on graph topology:
            // Once fundamentals completes → summary, risk, info become running
            if (agentName === 'fundamentals') {
              for (const parallel of PARALLEL_AGENTS) {
                if (!completedRef.current.has(parallel)) {
                  currentAgents[parallel] = {
                    ...currentAgents[parallel],
                    status: 'running',
                  };
                }
              }
            }

            // Once all three parallel agents complete → synthesis becomes running
            const allParallelDone = PARALLEL_AGENTS.every((a) =>
              completedRef.current.has(a)
            );
            if (allParallelDone && !completedRef.current.has('synthesis')) {
              currentAgents.synthesis = {
                ...currentAgents.synthesis,
                status: 'running',
              };
            }

            // Extract the final brief from synth_agent
            if (agentName === 'synthesis' && data[messageKey]?.final_output) {
              brief = data[messageKey].final_output;
              setFinalBrief(brief);
            }

            setAgents({ ...currentAgents });
          } catch {
            // Ignore malformed messages
          }
        };

        ws.onclose = () => {
          setIsRunning(false);
          wsRef.current = null;

          const result: AnalysisResult = {
            ticker,
            timestamp: new Date().toISOString(),
            agents: currentAgents,
            finalBrief: brief,
          };
          resolve(result);
        };

        ws.onerror = () => {
          setError('Connection failed — is the backend running on localhost:8000?');
          setIsRunning(false);
          wsRef.current = null;
          resolve(null);
        };
      } catch {
        setError('Failed to establish WebSocket connection.');
        setIsRunning(false);
        resolve(null);
      }
    });
  }, []);

  const clearError = useCallback(() => {
    setError(null);
  }, []);

  const loadResult = useCallback((result: AnalysisResult) => {
    setAgents(result.agents);
    setFinalBrief(result.finalBrief);
    setError(null);
    setIsRunning(false);
  }, []);

  const resetState = useCallback(() => {
    setAgents(createInitialAgents());
    setFinalBrief(null);
    setError(null);
    setIsRunning(false);
  }, []);

  return {
    agents,
    finalBrief,
    isRunning,
    error,
    runAnalysis,
    clearError,
    loadResult,
    resetState,
  };
}
