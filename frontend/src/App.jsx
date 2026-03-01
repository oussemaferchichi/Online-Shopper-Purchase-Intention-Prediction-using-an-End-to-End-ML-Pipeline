import { useState } from 'react';
import './index.css';
import Dashboard from './Dashboard';
import PredictionForm from './PredictionForm';
import ResultCard from './ResultCard';
import BatchTab from './BatchTab';

const TABS = [
  { id: 'dashboard', label: '📊 Dashboard' },
  { id: 'predict', label: '🔮 Predict' },
  { id: 'batch', label: '⚡ Batch' },
];

export default function App() {
  const [tab, setTab] = useState('dashboard');
  const [result, setResult] = useState(null);

  return (
    <div className="app-wrapper">
      {/* Header */}
      <header className="header">
        <div className="header-logo">
          <div className="logo-icon">🛒</div>
          <div>
            <div className="logo-text">ShopperAI</div>
            <div className="logo-sub">Purchase Intent Predictor</div>
          </div>
        </div>

        <nav className="header-nav">
          {TABS.map(t => (
            <button
              key={t.id}
              className={`nav-btn ${tab === t.id ? 'active' : ''}`}
              onClick={() => { setTab(t.id); if (t.id !== 'predict') setResult(null); }}
            >
              {t.label}
            </button>
          ))}
        </nav>

        <div className="status-dot">
          <span className="dot" /> API Live
        </div>
      </header>

      {/* Main content */}
      <main className="main">
        {tab === 'dashboard' && <Dashboard />}

        {tab === 'predict' && (
          <>
            <h2 className="section-title">Purchase Prediction</h2>
            <p className="section-sub">Fill in the session details and get an instant HistGradientBoosting prediction.</p>
            <div className="predict-layout">
              <PredictionForm onResult={r => { setResult(r); }} />
              <ResultCard result={result} />
            </div>
          </>
        )}

        {tab === 'batch' && <BatchTab />}
      </main>

      {/* Footer */}
      <footer className="footer">
        ShopperAI — Python for Data Science · ML Pipeline Project ·
        Powered by <strong>HistGradientBoosting</strong> + <strong>FastAPI</strong> + <strong>React</strong>
      </footer>
    </div>
  );
}
