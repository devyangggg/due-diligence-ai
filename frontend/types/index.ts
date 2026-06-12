export type AgentName = 'fundamentals' | 'summary' | 'risk' | 'info' | 'synthesis';

export type AgentStatus = 'idle' | 'pending' | 'running' | 'complete' | 'error';

export interface AgentState {
  status: AgentStatus;
  label: string;
  data: unknown;
}

export interface AnalysisResult {
  ticker: string;
  timestamp: string;
  agents: Record<AgentName, AgentState>;
  finalBrief: string | null;
}

export interface HistoryEntry {
  id: string;
  ticker: string;
  timestamp: string;
  result: AnalysisResult;
}

export const AGENT_CONFIG: { name: AgentName; label: string }[] = [
  { name: 'fundamentals', label: 'Fundamentals Analysis' },
  { name: 'summary', label: '10-K Summary' },
  { name: 'risk', label: 'Risk Assessment' },
  { name: 'info', label: 'Insider Information' },
  { name: 'synthesis', label: 'Due Diligence Synthesis' },
];

export function createInitialAgents(): Record<AgentName, AgentState> {
  return {
    fundamentals: { status: 'idle', label: 'Fundamentals Analysis', data: null },
    summary: { status: 'idle', label: '10-K Summary', data: null },
    risk: { status: 'idle', label: 'Risk Assessment', data: null },
    info: { status: 'idle', label: 'Insider Information', data: null },
    synthesis: { status: 'idle', label: 'Due Diligence Synthesis', data: null },
  };
}
