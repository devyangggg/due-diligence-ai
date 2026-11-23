export type AgentName = 'fundamentals' | 'summary' | 'risk' | 'info' | 'synthesis';
export type AgentStatus = 'idle' | 'pending' | 'running' | 'complete' | 'error';

export interface AgentState {
  status: AgentStatus;
  label: string;
  data: unknown;
}

export interface Citation {
  id: string;
  item: string;
  url: string;
  excerpt: string;
}

export interface Run {
  id: string;
  ticker: string;
  accession: string | null;
  status: 'running' | 'completed' | 'failed' | 'interrupted';
  startedAt: string;
  finishedAt: string | null;
  finalBrief: string | null;
  citations: Citation[];
  error: string | null;
}

export type AnalysisEvent =
  | { type: 'stage'; stage: AgentName; status: 'completed' }
  | { type: 'final'; run: Run }
  | { type: 'error'; run: Run; message: string };

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
