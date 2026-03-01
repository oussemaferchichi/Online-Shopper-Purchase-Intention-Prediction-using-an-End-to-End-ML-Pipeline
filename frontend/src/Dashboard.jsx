import { useEffect, useState } from 'react';
import axios from 'axios';
import API from './api';
import {
    RadarChart, Radar, PolarGrid, PolarAngleAxis,
    BarChart, Bar, XAxis, YAxis, Tooltip, Legend,
    ResponsiveContainer, Cell
} from 'recharts';

const MODEL_COLORS = {
    'Logistic Regression': '#64748b',
    'Decision Tree': '#f59e0b',
    'Random Forest': '#22d3ee',
    'HistGradientBoosting': '#6366f1',
};
const METRIC_COLORS = ['#6366f1', '#22d3ee', '#10b981', '#f59e0b', '#f43f5e'];

const Tooltip_ = ({ contentStyle = {} } = {}) => ({ contentStyle: { background: '#1e2433', border: '1px solid #2a3147', borderRadius: 8, fontSize: 12, ...contentStyle } });

export default function Dashboard() {
    const [info, setInfo] = useState(null);
    const [compare, setCompare] = useState(null);
    const [error, setError] = useState('');

    useEffect(() => {
        Promise.all([
            axios.get(`${API}/model-info`),
            axios.get(`${API}/models-comparison`),
        ])
            .then(([r1, r2]) => { setInfo(r1.data); setCompare(r2.data.models); })
            .catch(() => setError('Could not reach API. Make sure the backend is running.'));
    }, []);

    if (error) return <div className="error-banner">⚠ {error}</div>;
    if (!info || !compare) return <div style={{ color: 'var(--muted)', padding: '2rem' }}>Loading…</div>;

    const { metrics, dataset, training } = info;

    // Best-model metric cards
    const metricCards = [
        { label: 'Accuracy', value: `${(metrics.accuracy * 100).toFixed(1)}%`, cls: 'good', sub: 'Overall correctness' },
        { label: 'F1-Score', value: metrics.f1_score.toFixed(3), cls: 'good', sub: 'Precision × Recall balance' },
        { label: 'ROC-AUC', value: metrics.roc_auc.toFixed(3), cls: 'good', sub: 'Class separation ability' },
        { label: 'Precision', value: `${(metrics.precision * 100).toFixed(1)}%`, cls: 'warn', sub: 'Of predicted purchases' },
        { label: 'Recall', value: `${(metrics.recall * 100).toFixed(1)}%`, cls: 'warn', sub: 'Actual purchases caught' },
    ];

    // Single-model radar data
    const radarData = [
        { name: 'Accuracy', value: +(metrics.accuracy * 100).toFixed(1) },
        { name: 'Precision', value: +(metrics.precision * 100).toFixed(1) },
        { name: 'Recall', value: +(metrics.recall * 100).toFixed(1) },
        { name: 'F1', value: +(metrics.f1_score * 100).toFixed(1) },
        { name: 'ROC-AUC', value: +(metrics.roc_auc * 100).toFixed(1) },
    ];

    // Grouped bar chart data: one row per metric
    const groupedData = [
        { metric: 'Accuracy', ...Object.fromEntries(compare.map(m => [m.name, +(m.accuracy * 100).toFixed(1)])) },
        { metric: 'F1-Score', ...Object.fromEntries(compare.map(m => [m.name, +(m.f1_score * 100).toFixed(1)])) },
        { metric: 'ROC-AUC', ...Object.fromEntries(compare.map(m => [m.name, +(m.roc_auc * 100).toFixed(1)])) },
        { metric: 'Precision', ...Object.fromEntries(compare.map(m => [m.name, +(m.precision * 100).toFixed(1)])) },
        { metric: 'Recall', ...Object.fromEntries(compare.map(m => [m.name, +(m.recall * 100).toFixed(1)])) },
    ];

    const tooltipStyle = { background: '#1e2433', border: '1px solid #2a3147', borderRadius: 8, fontSize: 12 };

    return (
        <>
            <h2 className="section-title">Model Dashboard</h2>
            <p className="section-sub">{info.model_name} — {info.description}</p>

            {/* Best model metric cards */}
            <div className="metrics-grid">
                {metricCards.map(c => (
                    <div className="metric-card" key={c.label}>
                        <div className="metric-label">{c.label}</div>
                        <div className={`metric-value ${c.cls}`}>{c.value}</div>
                        <div className="metric-sub">{c.sub}</div>
                    </div>
                ))}
            </div>

            {/* ── All Models Comparison ───────────────────────────────────── */}
            <h2 className="section-title" style={{ marginTop: '2rem' }}>📊 All Models Comparison</h2>
            <p className="section-sub">4 models trained and tracked with MLflow — grouped by metric</p>

            {/* Grouped bar chart */}
            <div className="chart-card" style={{ marginBottom: '1.5rem' }}>
                <div className="chart-title">Performance by Metric (all 4 models)</div>
                <ResponsiveContainer width="100%" height={280}>
                    <BarChart data={groupedData} margin={{ top: 5, right: 20, left: -10, bottom: 5 }}>
                        <XAxis dataKey="metric" tick={{ fontSize: 11, fill: '#64748b' }} />
                        <YAxis domain={[50, 100]} tick={{ fontSize: 11, fill: '#64748b' }} unit="%" />
                        <Tooltip contentStyle={tooltipStyle} formatter={v => `${v}%`} />
                        <Legend wrapperStyle={{ fontSize: 12, color: '#94a3b8' }} />
                        {compare.map(m => (
                            <Bar key={m.name} dataKey={m.name} fill={MODEL_COLORS[m.name]} radius={[4, 4, 0, 0]} />
                        ))}
                    </BarChart>
                </ResponsiveContainer>
            </div>

            {/* Comparison table */}
            <div className="chart-card" style={{ marginBottom: '1.5rem', overflowX: 'auto' }}>
                <div className="chart-title">📋 Full Comparison Table</div>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.85rem' }}>
                    <thead>
                        <tr>
                            {['Model', 'Type', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'].map(h => (
                                <th key={h} style={{ padding: '0.5rem 0.75rem', textAlign: 'left', color: 'var(--muted)', borderBottom: '1px solid var(--border)', fontWeight: 600, fontSize: '0.75rem', textTransform: 'uppercase' }}>
                                    {h}
                                </th>
                            ))}
                        </tr>
                    </thead>
                    <tbody>
                        {compare.map((m, i) => (
                            <tr key={m.name} style={{ background: i % 2 === 0 ? 'var(--surface2)' : 'transparent' }}>
                                <td style={{ padding: '0.6rem 0.75rem', fontWeight: 600, color: MODEL_COLORS[m.name] }}>
                                    {m.name}
                                </td>
                                <td style={{ padding: '0.6rem 0.75rem', color: 'var(--muted)', fontSize: '0.78rem' }}>{m.type}</td>
                                {[m.accuracy, m.precision, m.recall, m.f1_score, m.roc_auc].map((v, j) => (
                                    <td key={j} style={{ padding: '0.6rem 0.75rem' }}>
                                        {(v * 100).toFixed(2)}%
                                    </td>
                                ))}
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>

            {/* Best model radar + info */}
            <div className="charts-grid">
                <div className="chart-card">
                    <div className="chart-title">🕸 Best Model Radar — HistGradientBoosting</div>
                    <ResponsiveContainer width="100%" height={220}>
                        <RadarChart data={radarData}>
                            <PolarGrid stroke="#2a3147" />
                            <PolarAngleAxis dataKey="name" tick={{ fontSize: 11, fill: '#64748b' }} />
                            <Radar dataKey="value" stroke="#6366f1" fill="#6366f1" fillOpacity={0.25} />
                            <Tooltip contentStyle={tooltipStyle} formatter={v => `${v}%`} />
                        </RadarChart>
                    </ResponsiveContainer>
                </div>

                <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                    <div className="info-card">
                        <h3>Dataset Info</h3>
                        {[
                            ['Total Samples', dataset.total_samples.toLocaleString()],
                            ['Features', dataset.features],
                            ['SMOTE Applied', dataset.smote_applied ? '✅ Yes' : '❌ No'],
                            ['Purchase Rate', dataset.purchase_rate],
                        ].map(([k, v]) => (
                            <div className="info-row" key={k}><span>{k}</span><span>{v}</span></div>
                        ))}
                    </div>
                    <div className="info-card">
                        <h3>Training Details</h3>
                        {[
                            ['Best Model', 'HistGradientBoosting'],
                            ['GridSearchCV', training.gridsearchcv ? '✅ Yes' : '❌ No'],
                            ['CV Folds', training.cv_folds],
                            ['CV Scoring', training.scoring.toUpperCase()],
                        ].map(([k, v]) => (
                            <div className="info-row" key={k}><span>{k}</span><span>{v}</span></div>
                        ))}
                    </div>
                </div>
            </div>
        </>
    );
}
