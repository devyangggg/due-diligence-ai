'use client';

interface ErrorBannerProps {
  message: string;
  onDismiss: () => void;
}

export default function ErrorBanner({ message, onDismiss }: ErrorBannerProps) {
  return (
    <div className="error-banner" role="alert">
      <div className="error-banner-content">
        <svg
          width="18"
          height="18"
          viewBox="0 0 18 18"
          fill="none"
          className="error-icon"
        >
          <circle cx="9" cy="9" r="8" stroke="currentColor" strokeWidth="1.5" />
          <path
            d="M9 5.5v4M9 12v.5"
            stroke="currentColor"
            strokeWidth="1.5"
            strokeLinecap="round"
          />
        </svg>
        <p className="error-message">{message}</p>
      </div>
      <button
        className="error-dismiss"
        onClick={onDismiss}
        aria-label="Dismiss error"
      >
        <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
          <path
            d="M4 4l8 8M12 4l-8 8"
            stroke="currentColor"
            strokeWidth="1.5"
            strokeLinecap="round"
          />
        </svg>
      </button>
    </div>
  );
}
