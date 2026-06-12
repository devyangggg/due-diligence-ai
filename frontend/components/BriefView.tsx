"use client";

import ReactMarkdown from "react-markdown";

interface BriefViewProps {
  finalBrief: string | null;
  isRunning: boolean;
}

export default function BriefView({ finalBrief, isRunning }: BriefViewProps) {
  // Not running and no brief — show nothing
  if (!isRunning && !finalBrief) return null;

  // Running but no brief yet — show loading state
  if (isRunning && !finalBrief) {
    return (
      <div className="brief-loading">
        <div className="brief-loading-inner">
          <span className="spinner spinner-lg" />
          <p className="brief-loading-text">Compiling due diligence brief…</p>
          <p className="brief-loading-subtext">
            The synthesis agent will produce a comprehensive report once all
            analyses are complete.
          </p>
        </div>
      </div>
    );
  }

  // Brief is ready
  {
    console.log(finalBrief);
  }
  return (
    <div className="brief-container">
      <div className="brief-header">
        <h2 className="brief-title">Due Diligence Brief</h2>
        <div className="brief-divider" />
      </div>
      <div className="brief-content prose">
        <ReactMarkdown>{finalBrief!}</ReactMarkdown>
      </div>
    </div>
  );
}
