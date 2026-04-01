from flask import Flask, render_template_string, request
import numpy as np
import pickle

app = Flask(__name__)

# ==============================
# LOAD XGBOOST MODEL
# ==============================

model = pickle.load(open("../xgboost/xgb_model.pkl", "rb"))

# ==============================
# HTML UI — PREMIUM CARDIO THEME
# ==============================

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CardioAI — Cardiac Risk Assessment</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Serif+Display:ital@0;1&display=swap" rel="stylesheet">

<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    font-family: 'DM Sans', sans-serif;
    background: #f8f7f4;
    color: #1a1a1a;
    min-height: 100vh;
  }

  /* ── TOP BAR ── */
  .topbar {
    background: #ffffff;
    border-bottom: 0.5px solid #e2e0db;
    padding: 14px 40px;
    display: flex;
    align-items: center;
    gap: 12px;
    position: sticky;
    top: 0;
    z-index: 100;
  }

  .logo-dot {
    width: 34px; height: 34px;
    border-radius: 50%;
    background: #c0392b;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
  }

  .logo-dot svg { width: 18px; height: 18px; fill: white; }

  .brand { font-size: 16px; font-weight: 600; color: #1a1a1a; letter-spacing: -0.3px; }
  .brand span { color: #c0392b; }

  .tagline {
    margin-left: auto;
    font-size: 12px;
    color: #aaa;
    letter-spacing: 0.4px;
  }

  /* ── LAYOUT ── */
  .main { max-width: 960px; margin: 0 auto; padding: 36px 24px 60px; }

  .hero { margin-bottom: 28px; }
  .hero h1 {
    font-family: 'DM Serif Display', serif;
    font-size: 30px;
    font-weight: 400;
    color: #1a1a1a;
    margin-bottom: 6px;
  }
  .hero p { font-size: 14px; color: #888; }

  .grid {
    display: grid;
    grid-template-columns: 1fr 310px;
    gap: 20px;
    align-items: start;
  }

  /* ── CARDS ── */
  .card {
    background: #ffffff;
    border: 0.5px solid #e2e0db;
    border-radius: 16px;
    padding: 26px;
    margin-bottom: 0;
  }

  .section-label {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 1.2px;
    color: #c0392b;
    text-transform: uppercase;
    margin-bottom: 16px;
  }

  /* ── FORM FIELDS ── */
  .fields {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 14px;
    margin-bottom: 4px;
  }

  .field { display: flex; flex-direction: column; gap: 5px; }

  .field label {
    font-size: 11px;
    font-weight: 500;
    color: #888;
    letter-spacing: 0.3px;
  }

  .field input,
  .field select {
    padding: 9px 12px;
    border: 0.5px solid #ddd;
    border-radius: 8px;
    font-size: 13px;
    font-family: 'DM Sans', sans-serif;
    color: #1a1a1a;
    background: #fafaf8;
    outline: none;
    transition: border-color 0.2s, background 0.2s;
    width: 100%;
  }

  .field input:focus,
  .field select:focus {
    border-color: #c0392b;
    background: #ffffff;
  }

  .field .hint { font-size: 10px; color: #ccc; margin-top: 2px; }

  .section-gap { margin-top: 22px; }

  /* ── ACTIONS ── */
  .actions { display: flex; gap: 10px; margin-top: 24px; }

  .btn-primary {
    flex: 1;
    padding: 12px;
    background: #c0392b;
    color: white;
    border: none;
    border-radius: 10px;
    font-size: 13px;
    font-weight: 600;
    font-family: 'DM Sans', sans-serif;
    cursor: pointer;
    letter-spacing: 0.3px;
    transition: background 0.2s, transform 0.1s;
  }
  .btn-primary:hover { background: #a93226; }
  .btn-primary:active { transform: scale(0.98); }

  .btn-secondary {
    padding: 12px 20px;
    background: #f0eeea;
    color: #555;
    border: none;
    border-radius: 10px;
    font-size: 13px;
    font-weight: 500;
    font-family: 'DM Sans', sans-serif;
    cursor: pointer;
    transition: background 0.2s;
  }
  .btn-secondary:hover { background: #e5e2dc; }

  /* ── SIDEBAR ── */
  .sidebar { display: flex; flex-direction: column; gap: 16px; }

  .risk-card {
    background: #ffffff;
    border: 0.5px solid #e2e0db;
    border-radius: 16px;
    padding: 26px 20px;
    text-align: center;
  }

  .risk-meter { margin: 0 auto 16px; width: 150px; height: 82px; }
  .risk-meter svg { overflow: visible; }

  .risk-title {
    font-size: 20px;
    font-weight: 600;
    margin-bottom: 4px;
    color: #1a1a1a;
  }

  .risk-desc { font-size: 12px; color: #999; margin-bottom: 14px; line-height: 1.5; }

  .risk-badge {
    display: inline-block;
    padding: 5px 18px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.4px;
  }

  .badge-low      { background: #eaf3de; color: #3B6D11; }
  .badge-moderate { background: #faeeda; color: #854F0B; }
  .badge-high     { background: #fcebeb; color: #A32D2D; }
  .badge-critical { background: #501313; color: #F7C1C1; }

  /* ── SUMMARY TABLE ── */
  .summary-card {
    background: #ffffff;
    border: 0.5px solid #e2e0db;
    border-radius: 16px;
    padding: 20px;
  }

  .summary-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 0;
    border-bottom: 0.5px solid #f0eeea;
  }
  .summary-row:last-child { border-bottom: none; }
  .summary-row .key { font-size: 12px; color: #999; }
  .summary-row .val { font-size: 12px; font-weight: 500; color: #1a1a1a; }

  /* ── DISCLAIMER ── */
  .disclaimer {
    background: #fafaf8;
    border: 0.5px solid #e2e0db;
    border-radius: 10px;
    padding: 14px;
    font-size: 11px;
    color: #aaa;
    line-height: 1.7;
  }
  .disclaimer strong { color: #c0392b; }

  /* ── ERROR ── */
  .error-box {
    background: #fcebeb;
    border: 0.5px solid #f09595;
    border-radius: 10px;
    padding: 12px 16px;
    font-size: 13px;
    color: #A32D2D;
    margin-top: 16px;
  }

  /* ── RESPONSIVE ── */
  @media (max-width: 680px) {
    .grid { grid-template-columns: 1fr; }
    .fields { grid-template-columns: 1fr; }
    .topbar { padding: 14px 20px; }
    .tagline { display: none; }
    .main { padding: 24px 16px 48px; }
  }
</style>

<!-- Auto-fill sample patients via JS -->
<script>
const PATIENTS = [
  {age:63,sex:1,cp:3,trestbps:145,chol:233,fbs:1,restecg:0,thalach:150,exang:0,oldpeak:2.3,slope:0,ca:0,thal:1},
  {age:52,sex:1,cp:0,trestbps:125,chol:212,fbs:0,restecg:1,thalach:168,exang:0,oldpeak:1.0,slope:2,ca:2,thal:2},
  {age:46,sex:0,cp:1,trestbps:105,chol:204,fbs:0,restecg:0,thalach:172,exang:0,oldpeak:0.0,slope:2,ca:0,thal:0},
  {age:58,sex:1,cp:2,trestbps:150,chol:270,fbs:0,restecg:0,thalach:111,exang:1,oldpeak:0.8,slope:1,ca:3,thal:2},
  {age:41,sex:0,cp:1,trestbps:130,chol:204,fbs:0,restecg:0,thalach:172,exang:0,oldpeak:1.4,slope:2,ca:0,thal:0},
  {age:67,sex:1,cp:0,trestbps:160,chol:286,fbs:0,restecg:0,thalach:108,exang:1,oldpeak:1.5,slope:1,ca:3,thal:2},
  {age:37,sex:1,cp:2,trestbps:130,chol:250,fbs:0,restecg:1,thalach:187,exang:0,oldpeak:3.5,slope:0,ca:0,thal:0},
  {age:55,sex:0,cp:1,trestbps:132,chol:342,fbs:0,restecg:0,thalach:166,exang:0,oldpeak:1.2,slope:2,ca:0,thal:0}
];
let _pidx = 0;

function simulatePatient() {
  const p = PATIENTS[_pidx % PATIENTS.length]; _pidx++;
  const fields = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal'];
  fields.forEach(f => { document.getElementById(f).value = p[f]; });
}
</script>

</head>
<body>

<!-- TOP BAR -->
<div class="topbar">
  <div class="logo-dot">
    <svg viewBox="0 0 24 24"><path d="M12 21.35l-1.45-1.32C5.4 15.36 2 12.28 2 8.5 2 5.42 4.42 3 7.5 3c1.74 0 3.41.81 4.5 2.09C13.09 3.81 14.76 3 16.5 3 19.58 3 22 5.42 22 8.5c0 3.78-3.4 6.86-8.55 11.54L12 21.35z"/></svg>
  </div>
  <div class="brand">Cardio<span>AI</span></div>
  <div class="tagline">Cardiac Risk Assessment System &bull; Research Use Only</div>
</div>

<!-- MAIN -->
<div class="main">
  <div class="hero">
    <h1>Cardiac Risk Assessment</h1>
    <p>Enter patient vitals to predict heart disease risk using our XGBoost classifier trained on the UCI Cleveland Heart Disease Dataset.</p>
  </div>

  <div class="grid">

    <!-- ── LEFT: FORM ── -->
    <div class="card">
      <form method="POST" id="riskForm">

        <!-- DEMOGRAPHICS -->
        <div class="section-label">Patient Demographics</div>
        <div class="fields">
          <div class="field">
            <label>Age (years)</label>
            <input id="age" name="age" type="number" placeholder="e.g. 54" min="20" max="100" value="{{ form.age }}" required>
            <span class="hint">Typical range: 29 – 77 years</span>
          </div>
          <div class="field">
            <label>Biological Sex</label>
            <select id="sex" name="sex" required>
              <option value="" disabled {{ 'selected' if not form.sex }}>Select...</option>
              <option value="1" {{ 'selected' if form.sex == '1' }}>Male</option>
              <option value="0" {{ 'selected' if form.sex == '0' }}>Female</option>
            </select>
          </div>
        </div>

        <!-- CHEST PAIN -->
        <div class="section-gap">
          <div class="section-label">Chest Pain &amp; Symptoms</div>
        </div>
        <div class="fields">
          <div class="field">
            <label>Chest Pain Type</label>
            <select id="cp" name="cp" required>
              <option value="" disabled {{ 'selected' if not form.cp }}>Select type...</option>
              <option value="0" {{ 'selected' if form.cp == '0' }}>Typical Angina</option>
              <option value="1" {{ 'selected' if form.cp == '1' }}>Atypical Angina</option>
              <option value="2" {{ 'selected' if form.cp == '2' }}>Non-Anginal Pain</option>
              <option value="3" {{ 'selected' if form.cp == '3' }}>Asymptomatic</option>
            </select>
            <span class="hint">Type of chest discomfort reported</span>
          </div>
          <div class="field">
            <label>Exercise-Induced Angina</label>
            <select id="exang" name="exang" required>
              <option value="" disabled {{ 'selected' if not form.exang }}>Select...</option>
              <option value="1" {{ 'selected' if form.exang == '1' }}>Yes — chest pain during exercise</option>
              <option value="0" {{ 'selected' if form.exang == '0' }}>No</option>
            </select>
          </div>
        </div>

        <!-- BLOOD METRICS -->
        <div class="section-gap">
          <div class="section-label">Blood Pressure &amp; Cholesterol</div>
        </div>
        <div class="fields">
          <div class="field">
            <label>Resting Blood Pressure (mmHg)</label>
            <input id="trestbps" name="trestbps" type="number" placeholder="e.g. 130" min="80" max="220" value="{{ form.trestbps }}">
            <span class="hint">Normal: 90–120 mmHg</span>
          </div>
          <div class="field">
            <label>Serum Cholesterol (mg/dL)</label>
            <input id="chol" name="chol" type="number" placeholder="e.g. 240" min="100" max="600" value="{{ form.chol }}">
            <span class="hint">Normal: &lt;200 mg/dL</span>
          </div>
          <div class="field">
            <label>Fasting Blood Sugar &gt;120 mg/dL</label>
            <select id="fbs" name="fbs" required>
              <option value="" disabled {{ 'selected' if not form.fbs }}>Select...</option>
              <option value="1" {{ 'selected' if form.fbs == '1' }}>Yes (diabetic range)</option>
              <option value="0" {{ 'selected' if form.fbs == '0' }}>No (normal)</option>
            </select>
          </div>
          <div class="field">
            <label>ST Depression — Oldpeak</label>
            <input id="oldpeak" name="oldpeak" type="number" placeholder="e.g. 1.5" step="0.1" min="0" max="10" value="{{ form.oldpeak }}">
            <span class="hint">ST depression induced by exercise vs rest</span>
          </div>
        </div>

        <!-- HEART RATE & ECG -->
        <div class="section-gap">
          <div class="section-label">Heart Rate &amp; ECG Findings</div>
        </div>
        <div class="fields">
          <div class="field">
            <label>Max Heart Rate Achieved (bpm)</label>
            <input id="thalach" name="thalach" type="number" placeholder="e.g. 150" min="60" max="220" value="{{ form.thalach }}">
            <span class="hint">During treadmill stress test</span>
          </div>
          <div class="field">
            <label>Resting ECG Result</label>
            <select id="restecg" name="restecg" required>
              <option value="" disabled {{ 'selected' if not form.restecg }}>Select...</option>
              <option value="0" {{ 'selected' if form.restecg == '0' }}>Normal</option>
              <option value="1" {{ 'selected' if form.restecg == '1' }}>ST-T Wave Abnormality</option>
              <option value="2" {{ 'selected' if form.restecg == '2' }}>Left Ventricular Hypertrophy</option>
            </select>
          </div>
          <div class="field">
            <label>ST Slope at Exercise Peak</label>
            <select id="slope" name="slope" required>
              <option value="" disabled {{ 'selected' if not form.slope }}>Select...</option>
              <option value="0" {{ 'selected' if form.slope == '0' }}>Upsloping</option>
              <option value="1" {{ 'selected' if form.slope == '1' }}>Flat</option>
              <option value="2" {{ 'selected' if form.slope == '2' }}>Downsloping</option>
            </select>
            <span class="hint">Shape of ST segment on ECG</span>
          </div>
          <div class="field">
            <label>Major Vessels Colored (0–3)</label>
            <select id="ca" name="ca" required>
              <option value="" disabled {{ 'selected' if not form.ca }}>Select...</option>
              <option value="0" {{ 'selected' if form.ca == '0' }}>0 vessels</option>
              <option value="1" {{ 'selected' if form.ca == '1' }}>1 vessel</option>
              <option value="2" {{ 'selected' if form.ca == '2' }}>2 vessels</option>
              <option value="3" {{ 'selected' if form.ca == '3' }}>3 vessels</option>
            </select>
            <span class="hint">Via fluoroscopy scan</span>
          </div>
        </div>

        <!-- THALASSEMIA -->
        <div class="section-gap">
          <div class="section-label">Thalassemia</div>
        </div>
        <div class="fields" style="grid-template-columns: 1fr;">
          <div class="field">
            <label>Thalassemia Blood Disorder Type</label>
            <select id="thal" name="thal" required>
              <option value="" disabled {{ 'selected' if not form.thal }}>Select...</option>
              <option value="0" {{ 'selected' if form.thal == '0' }}>Normal</option>
              <option value="1" {{ 'selected' if form.thal == '1' }}>Fixed Defect — no blood flow in part of heart</option>
              <option value="2" {{ 'selected' if form.thal == '2' }}>Reversible Defect — blood flow restored after rest</option>
            </select>
          </div>
        </div>

        <!-- ERROR -->
        {% if error %}
        <div class="error-box">⚠ {{ error }}</div>
        {% endif %}

        <!-- BUTTONS -->
        <div class="actions">
          <button class="btn-primary" type="submit">Analyze Risk</button>
          <button class="btn-secondary" type="button" onclick="simulatePatient()">Simulate Patient</button>
        </div>

      </form>
    </div>

    <!-- ── RIGHT: SIDEBAR ── -->
    <div class="sidebar">

      <!-- RISK RESULT -->
      <div class="risk-card">
        <div class="section-label" style="text-align:center; margin-bottom:20px;">Risk Score</div>

        <div class="risk-meter">
          <svg viewBox="0 0 150 82" width="150" height="82">
            <defs>
              <linearGradient id="mGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%"   stop-color="#3B6D11"/>
                <stop offset="35%"  stop-color="#EF9F27"/>
                <stop offset="70%"  stop-color="#E24B4A"/>
                <stop offset="100%" stop-color="#501313"/>
              </linearGradient>
            </defs>
            <!-- track -->
            <path d="M 10 72 A 65 65 0 0 1 140 72"
                  fill="none" stroke="#eeece7" stroke-width="10" stroke-linecap="round"/>
            <!-- filled arc -->
            <path id="mArc" d="M 10 72 A 65 65 0 0 1 140 72"
                  fill="none" stroke="url(#mGrad)" stroke-width="10" stroke-linecap="round"
                  stroke-dasharray="{{ arc_total }}" stroke-dashoffset="{{ arc_offset }}"/>
            <!-- needle dot -->
            <circle id="mDot" cx="{{ dot_x }}" cy="{{ dot_y }}" r="6" fill="#1a1a1a"/>
            <!-- percentage text -->
            <text x="75" y="56" text-anchor="middle"
                  font-size="22" font-weight="600"
                  font-family="DM Sans, sans-serif" fill="#1a1a1a">
              {{ pct_text }}
            </text>
            <text x="75" y="70" text-anchor="middle"
                  font-size="9" font-family="DM Sans, sans-serif" fill="#aaa">
              PROBABILITY
            </text>
          </svg>
        </div>

        <div class="risk-title">{{ risk_title }}</div>
        <div class="risk-desc">{{ risk_desc }}</div>

        {% if badge_class %}
        <span class="risk-badge {{ badge_class }}">{{ pct_text }} Probability</span>
        {% endif %}
      </div>

      <!-- INPUT SUMMARY -->
      {% if result %}
      <div class="summary-card">
        <div class="section-label">Input Summary</div>
        <div class="summary-row"><span class="key">Age</span><span class="val">{{ form.age }} yrs</span></div>
        <div class="summary-row"><span class="key">Sex</span><span class="val">{{ 'Male' if form.sex == '1' else 'Female' }}</span></div>
        <div class="summary-row"><span class="key">Blood Pressure</span><span class="val">{{ form.trestbps }} mmHg</span></div>
        <div class="summary-row"><span class="key">Cholesterol</span><span class="val">{{ form.chol }} mg/dL</span></div>
        <div class="summary-row"><span class="key">Max Heart Rate</span><span class="val">{{ form.thalach }} bpm</span></div>
        <div class="summary-row"><span class="key">ST Depression</span><span class="val">{{ form.oldpeak }}</span></div>
        <div class="summary-row"><span class="key">Vessels Colored</span><span class="val">{{ form.ca }}</span></div>
      </div>
      {% endif %}

      <!-- DISCLAIMER -->
      <div class="disclaimer">
        <strong>Clinical Disclaimer:</strong> This tool is intended for research and educational purposes only.
        It uses the <strong>UCI Cleveland Heart Disease Dataset</strong> with an XGBoost classifier.
        Do <strong>not</strong> use this output for actual medical diagnosis.
        Always consult a qualified cardiologist for clinical decisions.
      </div>

    </div><!-- /sidebar -->
  </div><!-- /grid -->
</div><!-- /main -->

</body>
</html>
"""

# ==============================
# HELPERS
# ==============================

CP_LABELS  = ['Typical Angina', 'Atypical Angina', 'Non-Anginal Pain', 'Asymptomatic']
THAL_LABELS = ['Normal', 'Fixed Defect', 'Reversible Defect']

import math

ARC_TOTAL = 204.2   # half-circle circumference for r=65 (π * 65)

def compute_arc(prob):
    """Return stroke-dashoffset and needle (cx, cy) for the SVG gauge."""
    offset = ARC_TOTAL * (1 - prob)
    # angle goes from -180° (left) to 0° (right) mapped to prob 0→1
    angle_deg = -180 + prob * 180
    angle_rad = math.radians(angle_deg)
    cx = 75 + 65 * math.cos(angle_rad)
    cy = 72 + 65 * math.sin(angle_rad)
    return round(offset, 1), round(cx, 1), round(cy, 1)


def classify(prob):
    pct = round(prob * 100, 1)
    if prob < 0.30:
        return "Low Risk",      "Cardiac risk within normal range. Maintain a healthy lifestyle.", "badge-low",      pct
    elif prob < 0.60:
        return "Moderate Risk", "Elevated risk detected. Further evaluation is recommended.",     "badge-moderate",  pct
    elif prob < 0.80:
        return "High Risk",     "Significant cardiac risk. Urgent cardiologist consultation advised.", "badge-high", pct
    else:
        return "Critical Risk", "Very high cardiac risk. Immediate medical attention required.",   "badge-critical",  pct


# ==============================
# ROUTE
# ==============================

@app.route("/", methods=["GET", "POST"])
def home():
    FIELDS = ['age','sex','cp','trestbps','chol','fbs','restecg',
              'thalach','exang','oldpeak','slope','ca','thal']

    # Default idle state
    ctx = dict(
        result       = False,
        error        = None,
        form         = {f: '' for f in FIELDS},
        risk_title   = "Awaiting Input",
        risk_desc    = "Enter patient data and click Analyze Risk.",
        badge_class  = "",
        pct_text     = "—",
        arc_total    = ARC_TOTAL,
        arc_offset   = ARC_TOTAL,   # fully empty arc
        dot_x        = 10,
        dot_y        = 72,
    )

    if request.method == "POST":
        # Persist form values so dropdowns/inputs stay filled on reload
        ctx["form"] = {f: request.form.get(f, '') for f in FIELDS}

        try:
            values = [float(request.form[f]) for f in FIELDS]
            arr    = np.array(values).reshape(1, -1)
            prob   = float(model.predict_proba(arr)[0][1])

            title, desc, badge, pct = classify(prob)
            offset, dx, dy          = compute_arc(prob)

            ctx.update(
                result      = True,
                risk_title  = title,
                risk_desc   = desc,
                badge_class = badge,
                pct_text    = f"{pct}%",
                arc_offset  = offset,
                dot_x       = dx,
                dot_y       = dy,
            )

        except KeyError as e:
            ctx["error"] = f"Missing field: {e}"
        except ValueError as e:
            ctx["error"] = f"Invalid value — please check all inputs. ({e})"
        except Exception as e:
            ctx["error"] = f"Prediction error: {e}"

    return render_template_string(HTML, **ctx)


# ==============================
# RUN
# ==============================

if __name__ == "__main__":
    app.run(debug=True)