import { useState } from 'react';
import axios from 'axios';
import API from './api';

const EXAMPLE_BATCH = JSON.stringify([
    {
        "Administrative": 0, "Administrative_Duration": 0, "Informational": 0,
        "Informational_Duration": 0, "ProductRelated": 35, "ProductRelated_Duration": 2500,
        "BounceRates": 0.01, "ExitRates": 0.03, "PageValues": 25.4, "SpecialDay": 0,
        "Month": "Nov", "OperatingSystems": 2, "Browser": 2, "Region": 1,
        "TrafficType": 2, "VisitorType": "Returning_Visitor", "Weekend": false
    },
    {
        "Administrative": 2, "Administrative_Duration": 60, "Informational": 1,
        "Informational_Duration": 30, "ProductRelated": 5, "ProductRelated_Duration": 300,
        "BounceRates": 0.2, "ExitRates": 0.2, "PageValues": 0, "SpecialDay": 0,
        "Month": "Feb", "OperatingSystems": 1, "Browser": 1, "Region": 3,
        "TrafficType": 4, "VisitorType": "New_Visitor", "Weekend": true
    }
], null, 2);

export default function BatchTab() {
    const [text, setText] = useState(EXAMPLE_BATCH);
    const [results, setResults] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');

    const submit = async () => {
        setError(''); setResults(null); setLoading(true);
        try {
            const parsed = JSON.parse(text);
            const res = await axios.post(`${API}/predict-batch`, parsed);
            setResults(res.data);
        } catch (e) {
            setError(e.response?.data?.detail || 'Invalid JSON or API error.');
        } finally { setLoading(false); }
    };

    return (
        <>
            <h2 className="section-title">Batch Prediction</h2>
            <p className="section-sub">Paste a JSON array of sessions below — all will be scored at once.</p>

            <div className="batch-card">
                {error && <div className="error-banner">⚠ {error}</div>}
                <textarea
                    className="batch-textarea"
                    value={text}
                    onChange={e => setText(e.target.value)}
                    spellCheck={false}
                />
                <button className="btn-batch" onClick={submit} disabled={loading}>
                    {loading ? <><span className="spinner" style={{ borderTopColor: '#000' }} /> Running…</> : '⚡ Run Batch'}
                </button>

                {results && (
                    <div className="batch-results">
                        <h3 style={{ marginBottom: '1rem', fontSize: '0.9rem', color: 'var(--muted)' }}>
                            {results.total} session{results.total !== 1 ? 's' : ''} processed
                        </h3>
                        {results.predictions.map((r, i) => {
                            const isPurch = r.prediction === 1;
                            return (
                                <div className="batch-row" key={i}>
                                    <span className="batch-index">#{i + 1}</span>
                                    <span className={`badge ${isPurch ? 'purchase' : 'no-purchase'}`}>
                                        {r.label}
                                    </span>
                                    <span className="batch-prob">
                                        Purchase: {(r.purchase_probability * 100).toFixed(1)}%
                                    </span>
                                </div>
                            );
                        })}
                    </div>
                )}
            </div>
        </>
    );
}
