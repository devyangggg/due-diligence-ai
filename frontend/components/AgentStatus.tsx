'use client';

import type { AgentName, AgentState, AgentStatus } from '@/types';
import { AGENT_CONFIG } from '@/types';

interface AgentStatusProps {
  agents: Record<AgentName, AgentState>;
}

function StatusIndicator({ status }: { status: AgentStatus }) {
  switch (status) {
    case 'idle':
      return (
        <span className="status-badge status-idle">
          <span className="status-dot status-dot-idle" />
          Idle
        </span>
      );
    case 'pending':
      return (
        <span className="status-badge status-pending">
          <span className="status-dot status-dot-pending" />
          Pending
        </span>
      );
    case 'running':
      return (
        <span className="status-badge status-running">
          <span className="status-dot status-dot-running" />
          Running…
        </span>
      );
    case 'complete':
      return (
        <span className="status-badge status-complete">
          <svg
            width="14"
            height="14"
            viewBox="0 0 14 14"
            fill="none"
            className="status-check"
          >
            <path
              d="M3 7l3 3 5-5"
              stroke="currentColor"
              strokeWidth="1.5"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
          Complete
        </span>
      );
    case 'error':
      return (
        <span className="status-badge status-error">
          <svg
            width="14"
            height="14"
            viewBox="0 0 14 14"
            fill="none"
            className="status-x"
          >
            <path
              d="M4 4l6 6M10 4l-6 6"
              stroke="currentColor"
              strokeWidth="1.5"
              strokeLinecap="round"
            />
          </svg>
          Failed
        </span>
      );
  }
}

function getStepNumber(name: AgentName): string {
  const map: Record<AgentName, string> = {
    fundamentals: '01',
    summary: '02',
    risk: '03',
    info: '04',
    synthesis: '05',
  };
  return map[name];
}

export default function AgentStatusPanel({ agents }: AgentStatusProps) {
  // Don't render anything if all agents are idle
  const allIdle = Object.values(agents).every((a) => a.status === 'idle');
  if (allIdle) return null;

  return (
    <div className="agent-status-panel">
      <h2 className="agent-status-title">Agent Progress</h2>
      <div className="agent-status-list">
        {AGENT_CONFIG.map(({ name, label }) => {
          const agent = agents[name];
          return (
            <div
              key={name}
              className={`agent-status-row ${
                agent.status === 'complete' ? 'agent-row-complete' : ''
              } ${agent.status === 'running' ? 'agent-row-running' : ''}`}
            >
              <div className="agent-status-left">
                <span className="agent-step-num">{getStepNumber(name)}</span>
                <span className="agent-label">{label}</span>
              </div>
              <StatusIndicator status={agent.status} />
            </div>
          );
        })}
      </div>
    </div>
  );
}
