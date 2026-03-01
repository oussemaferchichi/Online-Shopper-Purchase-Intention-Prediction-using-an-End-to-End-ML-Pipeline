import { useState } from 'react';
import axios from 'axios';
import API from './api';

const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'June', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
const VISITOR_TYPES = ['Returning_Visitor', 'New_Visitor', 'Other'];

const DEFAULT = {
    Administrative: 0, Administrative_Duration: 0,
    Informational: 0, Informational_Duration: 0,
    ProductRelated: 35, ProductRelated_Duration: 2500,
    BounceRates: 0.01, ExitRates: 0.03,
    PageValues: 25.4, SpecialDay: 0.0,
    Month: 'Nov', OperatingSystems: 2, Browser: 2,
    Region: 1, TrafficType: 2,
    VisitorType: 'Returning_Visitor', Weekend: false,
};

function Field({ label, name, type = 'number', value, onChange, options, step }) {
    if (type === 'select') return (
        <div className="form-field">
            <label>{label}</label>
            <select name={name} value={value} onChange={onChange}>
                {options.map(o => <option key={o} value={o}>{o}</option>)}
            </select>
        </div>
    );
    if (type === 'checkbox') return (
        <div className="form-field checkbox-field">
            <input type="checkbox" id={name} name={name} checked={value} onChange={onChange} />
            <label htmlFor={name}>{label}</label>
        </div>
    );
    return (
        <div className="form-field">
            <label>{label}</label>
            <input type="number" name={name} value={value} step={step || 'any'} min={0} onChange={onChange} />
        </div>
    );
}

export default function PredictionForm({ onResult }) {
    const [form, setForm] = useState(DEFAULT);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');

    const handle = e => {
        const { name, value, type, checked } = e.target;
        setForm(f => ({ ...f, [name]: type === 'checkbox' ? checked : (type === 'number' ? Number(value) : value) }));
    };

    const submit = async e => {
        e.preventDefault();
        setLoading(true); setError('');
        try {
            const res = await axios.post(`${API}/predict`, form);
            onResult(res.data);
        } catch {
            setError('Prediction failed. Is the API running?');
        } finally { setLoading(false); }
    };

    return (
        <div className="form-card">
            {error && <div className="error-banner">⚠ {error}</div>}
            <form onSubmit={submit}>
                {/* Page behaviour */}
                <div className="form-group-title">📄 Page Behaviour</div>
                <div className="form-grid">
                    <Field label="Administrative Pages" name="Administrative" value={form.Administrative} onChange={handle} />
                    <Field label="Admin Duration (s)" name="Administrative_Duration" value={form.Administrative_Duration} onChange={handle} />
                    <Field label="Informational Pages" name="Informational" value={form.Informational} onChange={handle} />
                    <Field label="Info Duration (s)" name="Informational_Duration" value={form.Informational_Duration} onChange={handle} />
                    <Field label="Product Related Pages" name="ProductRelated" value={form.ProductRelated} onChange={handle} />
                    <Field label="Product Duration (s)" name="ProductRelated_Duration" value={form.ProductRelated_Duration} onChange={handle} />
                </div>

                {/* Engagement rates */}
                <div className="form-group-title">📈 Engagement Rates</div>
                <div className="form-grid">
                    <Field label="Bounce Rate (0–1)" name="BounceRates" value={form.BounceRates} onChange={handle} step="0.01" />
                    <Field label="Exit Rate (0–1)" name="ExitRates" value={form.ExitRates} onChange={handle} step="0.01" />
                    <Field label="Page Value" name="PageValues" value={form.PageValues} onChange={handle} step="0.1" />
                    <Field label="Special Day (0–1)" name="SpecialDay" value={form.SpecialDay} onChange={handle} step="0.1" />
                </div>

                {/* Visitor profile */}
                <div className="form-group-title">👤 Visitor Profile</div>
                <div className="form-grid">
                    <Field label="Month" name="Month" type="select" value={form.Month} onChange={handle} options={MONTHS} />
                    <Field label="Visitor Type" name="VisitorType" type="select" value={form.VisitorType} onChange={handle} options={VISITOR_TYPES} />
                    <Field label="OS" name="OperatingSystems" value={form.OperatingSystems} onChange={handle} />
                    <Field label="Browser" name="Browser" value={form.Browser} onChange={handle} />
                    <Field label="Region" name="Region" value={form.Region} onChange={handle} />
                    <Field label="Traffic Type" name="TrafficType" value={form.TrafficType} onChange={handle} />
                </div>
                <Field label="Weekend visit?" name="Weekend" type="checkbox" value={form.Weekend} onChange={handle} />

                <button className="btn-predict" type="submit" disabled={loading}>
                    {loading ? <><span className="spinner" /> Predicting…</> : '🔮 Predict Purchase Intent'}
                </button>
            </form>
        </div>
    );
}
