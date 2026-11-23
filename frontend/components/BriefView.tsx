'use client';

import ReactMarkdown from 'react-markdown';
import type { Run } from '@/types';

interface BriefViewProps {
  run: Run | null;
  isRunning: boolean;
}

export default function BriefView({ run, isRunning }: BriefViewProps) {
  if (!isRunning && (!run || run.status !== 'completed' || !run.finalBrief)) return null;
  if (isRunning) {
    return (
      <div className="brief-loading">
        <div className="brief-loading-inner">
          <span className="spinner spinner-lg" />
          <p className="brief-loading-text">Compiling due diligence brief…</p>
          <p className="brief-loading-subtext">The synthesis agent will produce a comprehensive report once all analyses are complete.</p>
        </div>
      </div>
    );
  }

  const citations = new Map(run!.citations.map((citation) => [citation.id, citation]));
  const citedIds = Array.from(run!.finalBrief!.matchAll(/\[([A-Za-z0-9][A-Za-z0-9._:-]*)\]/g), (match) => match[1]);
  const unknownIds = [...new Set(citedIds.filter((id) => !citations.has(id) && /[-_:]/.test(id)))];

  return (
    <div className="brief-container">
      <div className="brief-header">
        <h2 className="brief-title">Due Diligence Brief</h2>
        <div className="brief-divider" />
      </div>
      <div className="brief-content prose">
        <ReactMarkdown components={{
          a: ({ href, children, ...props }) => {
            const id = href?.startsWith('#') ? decodeURIComponent(href.slice(1)) : null;
            if (id && citations.has(id)) return <a href={`#citation-${id}`} {...props}>{children}</a>;
            if (id) return <span title="Unknown citation ID">[Unverified citation: {id}]</span>;
            return <a href={href} {...props}>{children}</a>;
          },
        }}>{run!.finalBrief!.replace(/\[([A-Za-z0-9][A-Za-z0-9._:-]*)\]/g, (match, id: string) =>
          citations.has(id) ? `[${id}](#${id})` : match
        )}</ReactMarkdown>
        {unknownIds.length > 0 && <p>Unverified citation IDs: {unknownIds.join(', ')}</p>}
        {run!.citations.length > 0 && (
          <section aria-label="SEC filing evidence">
            <h3>Source evidence</h3>
            <ul>
              {run!.citations.map((citation) => (
                <li key={citation.id} id={`citation-${citation.id}`}>
                  <a href={citation.url} target="_blank" rel="noopener noreferrer">{citation.id} — {citation.item} (SEC filing)</a>
                  <blockquote>{citation.excerpt}</blockquote>
                </li>
              ))}
            </ul>
          </section>
        )}
      </div>
    </div>
  );
}
