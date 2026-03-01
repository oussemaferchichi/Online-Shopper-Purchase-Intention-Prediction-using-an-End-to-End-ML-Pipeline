import { useEffect, useState } from 'react';
import axios from 'axios';
import API from './api';
import {
    RadarChart, Radar, PolarGrid, PolarAngleAxis,
    BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell
} from 'recharts';

const COLORS = ['#6366f1', '#22d3ee', '#10b981', '#f59e0b', '#f43f5e'];

export default function Dashboard() {
    const [info, setInfo] = useState(null);
    const [error, setError] = useState('');

    useEffect(() => {
        axios.get(`${API}/model-info`)
            .then(r => setInfo(r.data))
            .catch(() => setError('Could not reach API. Make sure the backend is running.'));
    }, []);

    if (error) return <div className="error-banner">⚠ {error}</div>;
    if (!info) return <div style={{ color: 'var(--muted)', padding: '2rem' }}>Loading model info…</div>;

    const { metrics, dataset, training } = info;

    const metricCards = [
        { label: 'Accuracy', value: `${(metrics.accuracy * 100).toFixed(1)}%`, cls: 'good', sub: 'Overall correctness' },
        { label: 'F1-Score', value: metrics.f1_score.toFixed(3), cls: 'good', sub: 'Precision × Recall balance' },
        { label: 'ROC-AUC', value: metrics.roc_auc.toFixed(3), cls: 'good', sub: 'Class separation ability' },
        { label: 'Precision', value: `${(metrics.precision * 100).toFixed(1)}%`, cls: 'warn', sub: 'Of predicted purchases' },
        { label: 'Recall', value: `${(metrics.recall * 100).toFixed(1)}%`, cls: 'warn', sub: 'Of actual purchases caught' },
    ];

    const barData = [
        { name: 'Accuracy', value: +(metrics.accuracy * 100).toFixed(1) },
        { name: 'Precision', value: +(metrics.precision * 100).toFixed(1) },
        { name: 'Recall', value: +(metrics.recall * 100).toFixed(1) },
        { name: 'F1', value: +(metrics.f1_score * 100).toFixed(1) },
        { name: 'ROC-AUC', value: +(metrics.roc_auc * 100).toFixed(1) },
    ];

    const radarData = barData;

    return (
        <>
            <h2 className="section-title">Model Dashboard</h2>
            <p className="section-sub">{info.model_name} — {info.description}</p>

            {/* Metric cards */}
            <div className="metrics-grid">
                {metricCards.map(c => (
                    <div className="metric-card" key={c.label}>
                        <div className="metric-label">{c.label}</div>
                        <div className={`metric-value ${c.cls}`}>{c.value}</div>
                        <div className="metric-sub">{c.sub}</div>
                    </div>
                ))}
            </div>

            {/* Charts */}
            <div className="charts-grid">
                <div className="chart-card">
                    <div className="chart-title">📊 Metrics Overview (Bar)</div>
                    <ResponsiveContainer width="100%" height={220}>
                        <BarChart data={barData} margin={{ top: 5, right: 10, left: -10, bottom: 5 }}>
                            <XAxis dataKey="name" tick={{ fontSize: 11, fill: '#64748b' }} />
                            <YAxis domain={[0, 100]} tick={{ fontSize: 11, fill: '#64748b' }} />
                            <Tooltip
                                contentStyle={{ background: '#1e2433', border: '1px solid #2a3147', borderRadius: 8, fontSize: 12 }}
                                formatter={v => `${v}%`}
                            />
                            <Bar dataKey="value" radius={[6, 6, 0, 0]}>
                                {barData.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                            </Bar>
                        </BarChart>
                    </ResponsiveContainer>
                </div>

                <div className="chart-card">
                    <div className="chart-title">🕸 Metrics Radar</div>
                    <ResponsiveContainer width="100%" height={220}>
                        <RadarChart data={radarData}>
                            <PolarGrid stroke="#2a3147" />
                            <PolarAngleAxis dataKey="name" tick={{ fontSize: 11, fill: '#64748b' }} />
                            <Radar dataKey="value" stroke="#6366f1" fill="#6366f1" fillOpacity={0.25} />
                            <Tooltip
                                contentStyle={{ background: '#1e2433', border: '1px solid #2a3147', borderRadius: 8, fontSize: 12 }}
                                formatter={v => `${v}%`}
                            />
                        </RadarChart>
                    </ResponsiveContainer>
                </div>
            </div>

            {/* Info tables */}
            <div className="info-grid">
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
                        ['Model', info.model_name.split('(')[0].trim()],
                        ['GridSearchCV', training.gridsearchcv ? '✅ Yes' : '❌ No'],
                        ['CV Folds', training.cv_folds],
                        ['CV Scoring', training.scoring.toUpperCase()],
                    ].map(([k, v]) => (
                        <div className="info-row" key={k}><span>{k}</span><span>{v}</span></div>
                    ))}
                </div>
            </div>
        </>
    );
}
