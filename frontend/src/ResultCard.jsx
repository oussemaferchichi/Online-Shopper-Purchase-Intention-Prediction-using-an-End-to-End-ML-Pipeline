export default function ResultCard({ result }) {
    if (!result) return (
        <div className="placeholder-card">
            <div className="ph-icon">🎯</div>
            <p>Fill in the form on the left<br />and click <strong>Predict</strong> to see results.</p>
        </div>
    );

    const isPurchase = result.prediction === 1;
    const pct = (result.purchase_probability * 100).toFixed(1);
    const noPct = (result.no_purchase_probability * 100).toFixed(1);

    return (
        <div className="result-card">
            <div className="result-icon">{isPurchase ? '🛒' : '🚪'}</div>
            <div className={`result-label ${isPurchase ? 'purchase' : 'no-purchase'}`}>
                {result.label}
            </div>
            <div className="result-prob">
                Model confidence from HistGradientBoosting (GridSearchCV tuned)
            </div>

            <div className="prob-bar-wrap">
                <div className="prob-bar-label">
                    <span>🟢 Purchase</span>
                    <span>{pct}%</span>
                </div>
                <div className="prob-bar-bg">
                    <div className="prob-bar-fill green" style={{ width: `${pct}%` }} />
                </div>
            </div>

            <div className="prob-bar-wrap">
                <div className="prob-bar-label">
                    <span>🔴 No Purchase</span>
                    <span>{noPct}%</span>
                </div>
                <div className="prob-bar-bg">
                    <div className="prob-bar-fill red" style={{ width: `${noPct}%` }} />
                </div>
            </div>

            <div className="prob-value" style={{ color: isPurchase ? 'var(--success)' : 'var(--danger)' }}>
                {isPurchase ? pct : noPct}%
            </div>
            <div style={{ fontSize: '0.78rem', color: 'var(--muted)' }}>
                {isPurchase ? 'purchase probability' : 'pass probability'}
            </div>
        </div>
    );
}
