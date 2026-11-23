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

  const citationNumbers = new Map(run!.citations.map((citation, index) => [citation.id, index + 1] as const));
  const citedIds = Array.from(run!.finalBrief!.matchAll(/\[([A-Za-z0-9][A-Za-z0-9._:-]*)\]/g), (match) => match[1]);
  const unknownIds = [...new Set(citedIds.filter((id) => !citationNumbers.has(id) && /[-_:]/.test(id)))];

  return (
    <div className="brief-container">
      <div className="brief-header">
        <h2 className="brief-title">Due Diligence Brief</h2>
        <div className="brief-divider" />
      </div>
      <div className="brief-content prose">
        <ReactMarkdown components={{
          a: ({ href, children }) => {
            const id = href?.startsWith('#') ? decodeURIComponent(href.slice(1)) : null;
            if (id && citationNumbers.has(id)) {
              return <a className="source-link" href={`#citation-${id}`}>Source {citationNumbers.get(id)}</a>;
            }
            if (id) return <span title="Unknown citation ID">[Unverified citation: {id}]</span>;

            const sourceId = typeof children === 'string' && children.startsWith('source ')
              ? children.slice('source '.length)
              : null;
            const number = sourceId && citationNumbers.get(sourceId);
            if (number && href === run!.citations[number - 1].url) {
              const citation = run!.citations[number - 1];
              return (
                <a className="source-link" href={href} target="_blank" rel="noopener noreferrer"
                  aria-label={`Source ${number}: ${citation.item} in SEC filing`}>
                  Source {number}
                </a>
              );
            }
            return <a href={href}>{children}</a>;
          },
        }}>{run!.finalBrief!.replace(/\[([A-Za-z0-9][A-Za-z0-9._:-]*)\]/g, (match, id: string) =>
          citationNumbers.has(id) ? `[${id}](#${id})` : match
        )}</ReactMarkdown>
        {unknownIds.length > 0 && <p>Unverified citation IDs: {unknownIds.join(', ')}</p>}
        {run!.citations.length > 0 && (
          <section aria-label="SEC filing evidence">
            <h3>Source evidence</h3>
            <ul>
              {run!.citations.map((citation, index) => (
                <li key={citation.id} id={`citation-${citation.id}`}>
                  <a className="source-link" href={citation.url} target="_blank" rel="noopener noreferrer">
                    Source {index + 1} — {citation.item} (SEC filing)
                  </a>
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
