import {
  CheckCircle2,
  AlertTriangle,
  RefreshCw,
  RotateCcw,
  ShieldCheck,
  Brain,
  Heart,
  TrendingDown
} from 'lucide-react';
import './ResultCard.css';

function ResultCard({ result, onPredictAgain, onReset }) {
  const {
    prediction,
    prediction_code,
    probability,
    confidence,
    model: modelUsed = 'Random Forest'
  } = result || {};

  // Use ONLY prediction_code from backend — never string-match the label.
  // "No Cardiovascular Disease Detected" contains "detected" which
  // would falsely trigger a positive result if we string-match.
  const isPositive = prediction_code === 1;

  // Risk % = probability of having disease (class 1)
  const riskVal = probability !== undefined && probability !== null ? Number(probability) : null;
  const riskPercent =
    riskVal !== null
      ? (riskVal <= 1.0 && riskVal >= 0 ? riskVal * 100 : riskVal)
      : null;

  // Model confidence in its own prediction
  const confVal = confidence !== undefined && confidence !== null ? Number(confidence) : null;
  const confDisplay = confVal !== null ? confVal.toFixed(1) : null;

  const riskDisplay = riskPercent !== null ? riskPercent.toFixed(1) : null;

  // Gauge math – full circle circumference at r=38 is 2π×38 ≈ 238.76
  const CIRC = 238.76;
  const gaugeFill = riskPercent !== null
    ? CIRC - (CIRC * Math.min(Math.max(riskPercent, 0), 100)) / 100
    : CIRC;

  return (
    <div
      className={`rc ${isPositive ? 'rc--positive' : 'rc--negative'}`}
      role="region"
      aria-label="Prediction result"
    >
      {/* ── Header Status Strip ───────────────────────── */}
      <div className="rc-header">
        <div className={`rc-status-badge ${isPositive ? 'rc-status-badge--danger' : 'rc-status-badge--safe'}`}>
          {isPositive
            ? <AlertTriangle size={14} />
            : <CheckCircle2 size={14} />}
          <span>{isPositive ? 'HIGH CARDIOVASCULAR RISK' : 'LOW CARDIOVASCULAR RISK'}</span>
          <span className="rc-status-dot" />
        </div>

        <div className="rc-model-tag">
          <Brain size={13} />
          <span>{modelUsed}</span>
        </div>
      </div>

      {/* ── Main Body ─────────────────────────────────── */}
      <div className="rc-body">
        {/* Left: Headline + text */}
        <div className="rc-text">
          <div className={`rc-icon-wrap ${isPositive ? 'rc-icon-wrap--danger' : 'rc-icon-wrap--safe'}`}>
            {isPositive ? <Heart size={28} /> : <TrendingDown size={28} />}
          </div>

          <h2 className="rc-headline">
            {isPositive ? 'Cardiovascular Disease Detected' : 'No Cardiovascular Disease Detected'}
          </h2>

          <p className="rc-description">
            {isPositive
              ? `The ${modelUsed} model found elevated risk markers — including blood pressure, age, cholesterol, and lifestyle indicators — consistent with cardiovascular disease patterns in the training data.`
              : `The ${modelUsed} model evaluated all 11 clinical markers and found the health profile falls within normal, low-risk parameters. No significant cardiovascular risk factors were identified.`}
          </p>

          <div className="rc-meta-row">
            <div className="rc-meta-chip">
              <span className="rc-meta-lbl">PREDICTION CLASS</span>
              <span className="rc-meta-val">{isPositive ? 'Positive — Class 1' : 'Negative — Class 0'}</span>
            </div>
            {confDisplay !== null && (
              <div className="rc-meta-chip">
                <span className="rc-meta-lbl">MODEL CONFIDENCE</span>
                <span className="rc-meta-val">{confDisplay}%</span>
              </div>
            )}
          </div>
        </div>

        {/* Right: Risk Gauge */}
        {riskDisplay !== null && (
          <div className="rc-gauge-wrap">
            <div className="rc-gauge-label-top">DISEASE RISK</div>
            <div className="rc-gauge">
              <svg viewBox="0 0 100 100" className="rc-gauge-svg">
                <defs>
                  <linearGradient id="rcGradDanger" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" stopColor="#F87171" />
                    <stop offset="100%" stopColor="#DC2626" />
                  </linearGradient>
                  <linearGradient id="rcGradSafe" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" stopColor="#34D399" />
                    <stop offset="100%" stopColor="#059669" />
                  </linearGradient>
                  <filter id="rcGlow">
                    <feGaussianBlur stdDeviation="2.5" result="blur" />
                    <feMerge>
                      <feMergeNode in="blur" />
                      <feMergeNode in="SourceGraphic" />
                    </feMerge>
                  </filter>
                </defs>
                {/* Track */}
                <circle
                  cx="50" cy="50" r="38"
                  fill="none"
                  stroke="rgba(255,255,255,0.06)"
                  strokeWidth="9"
                />
                {/* Fill */}
                <circle
                  cx="50" cy="50" r="38"
                  fill="none"
                  stroke={isPositive ? 'url(#rcGradDanger)' : 'url(#rcGradSafe)'}
                  strokeWidth="9"
                  strokeLinecap="round"
                  strokeDasharray={CIRC}
                  strokeDashoffset={gaugeFill}
                  transform="rotate(-90 50 50)"
                  filter="url(#rcGlow)"
                  style={{ transition: 'stroke-dashoffset 1.2s cubic-bezier(0.4,0,0.2,1)' }}
                />
              </svg>

              <div className="rc-gauge-inner">
                <span className={`rc-gauge-num ${isPositive ? 'rc-gauge-num--danger' : 'rc-gauge-num--safe'}`}>
                  {riskDisplay}%
                </span>
                <span className="rc-gauge-sub">PROBABILITY</span>
              </div>
            </div>

            <div className={`rc-risk-level ${isPositive ? 'rc-risk-level--danger' : 'rc-risk-level--safe'}`}>
              {isPositive ? '⚠ Elevated Risk' : '✓ Low Risk'}
            </div>
          </div>
        )}
      </div>

      {/* ── Stat Bar ──────────────────────────────────── */}
      <div className="rc-stat-bar">
        <div className="rc-stat">
          <span className="rc-stat-lbl">Prediction</span>
          <span className={`rc-stat-val ${isPositive ? 'rc-stat-val--danger' : 'rc-stat-val--safe'}`}>
            {isPositive ? 'Disease Present' : 'Disease Absent'}
          </span>
        </div>
        {riskDisplay !== null && (
          <div className="rc-stat">
            <span className="rc-stat-lbl">Risk Score</span>
            <span className={`rc-stat-val ${isPositive ? 'rc-stat-val--danger' : 'rc-stat-val--safe'}`}>
              {riskDisplay}%
            </span>
          </div>
        )}
        {confDisplay !== null && (
          <div className="rc-stat">
            <span className="rc-stat-lbl">Confidence</span>
            <span className="rc-stat-val">{confDisplay}%</span>
          </div>
        )}
        <div className="rc-stat">
          <span className="rc-stat-lbl">Model Used</span>
          <span className="rc-stat-val">{modelUsed}</span>
        </div>
      </div>

      {/* ── Actions ───────────────────────────────────── */}
      <div className="rc-actions">
        <button onClick={onPredictAgain} className="btn btn--primary">
          <RefreshCw size={15} />
          <span>Run Another Prediction</span>
        </button>
        <button onClick={onReset} className="btn btn--outline">
          <RotateCcw size={15} />
          <span>Reset Form</span>
        </button>
      </div>

      {/* ── Disclaimer ────────────────────────────────── */}
      <div className="rc-disclaimer">
        <ShieldCheck size={14} className="rc-disclaimer-icon" />
        <p>
          <strong>Educational ML Output:</strong> This result is generated by an algorithmic model trained on a public dataset. It is not a clinical diagnosis. Always consult a certified healthcare professional.
        </p>
      </div>
    </div>
  );
}

export default ResultCard;
