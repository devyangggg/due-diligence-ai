'use client';

import type { HistoryEntry } from '@/types';

interface SidebarProps {
  entries: HistoryEntry[];
  activeId: string | null;
  isOpen: boolean;
  onClose: () => void;
  onSelect: (entry: HistoryEntry) => void;
  onNewAnalysis: () => void;
  onClearHistory: () => void;
}

export default function Sidebar({
  entries,
  activeId,
  isOpen,
  onClose,
  onSelect,
  onNewAnalysis,
  onClearHistory,
}: SidebarProps) {
  const handleSelect = (entry: HistoryEntry) => {
    onSelect(entry);
    onClose();
  };

  const handleNewAnalysis = () => {
    onNewAnalysis();
    onClose();
  };

  return (
    <>
      {/* Backdrop overlay — visible on mobile when drawer is open */}
      <div
        className={`sidebar-backdrop ${isOpen ? 'sidebar-backdrop-visible' : ''}`}
        onClick={onClose}
        aria-hidden="true"
      />

      <aside className={`sidebar ${isOpen ? 'sidebar-open' : ''}`}>
        {/* Close button — mobile only */}
        <button
          className="sidebar-close-btn"
          onClick={onClose}
          aria-label="Close menu"
        >
          <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
            <path
              d="M5 5l10 10M15 5L5 15"
              stroke="currentColor"
              strokeWidth="1.5"
              strokeLinecap="round"
            />
          </svg>
        </button>

        {/* Brand */}
        <div className="sidebar-brand">
          <h1 className="sidebar-title">Due Diligence AI</h1>
          <p className="sidebar-subtitle">Multi-Agent Research</p>
        </div>

        {/* New Analysis Button */}
        <button
          id="new-analysis-btn"
          className="new-analysis-btn"
          onClick={handleNewAnalysis}
        >
          <svg
            width="16"
            height="16"
            viewBox="0 0 16 16"
            fill="none"
            xmlns="http://www.w3.org/2000/svg"
          >
            <path
              d="M8 3v10M3 8h10"
              stroke="currentColor"
              strokeWidth="1.5"
              strokeLinecap="round"
            />
          </svg>
          New Analysis
        </button>

        {/* History */}
        <div className="sidebar-section">
          <div className="sidebar-section-header">
            <span className="sidebar-section-label">History</span>
            {entries.length > 0 && (
              <button
                className="clear-history-btn"
                onClick={onClearHistory}
              >
                Clear
              </button>
            )}
          </div>

          {entries.length === 0 ? (
            <p className="sidebar-empty">No analyses yet</p>
          ) : (
            <ul className="history-list">
              {entries.map((entry) => {
                const date = new Date(entry.timestamp);
                const formatted = date.toLocaleDateString('en-US', {
                  month: 'short',
                  day: 'numeric',
                  hour: '2-digit',
                  minute: '2-digit',
                });

                return (
                  <li key={entry.id}>
                    <button
                      className={`history-item ${
                        activeId === entry.id ? 'history-item-active' : ''
                      }`}
                      onClick={() => handleSelect(entry)}
                    >
                      <span className="history-ticker">{entry.ticker}</span>
                      <span className="history-time">{formatted}</span>
                    </button>
                  </li>
                );
              })}
            </ul>
          )}
        </div>
      </aside>
    </>
  );
}
