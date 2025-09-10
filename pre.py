import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ===============================
# Load Model & (optional) Scaler
# ===============================
try:
    data = joblib.load("diabetes_model.pkl")
    if isinstance(data, dict) and "model" in data:
        model = data["model"]
        scaler = data.get("scaler", None)
    else:
        model = data
        scaler = None
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# ===============================
# Page & Theme
# ===============================
st.set_page_config(page_title="Diabetes Prediction", layout="wide")

st.markdown("""
<style>
.stApp {
  background: linear-gradient(135deg, #f5f7fa, #c3cfe2);
  font-family: 'Segoe UI', sans-serif;
  color: #111;
}
.sidebar-card {
  background: linear-gradient(135deg, #e6e6ff, #d7c9ff);
  border-radius: 14px;
  padding: 16px;
  color: #111;
  text-align: center;
  box-shadow: 0 4px 16px rgba(0,0,0,0.12);
  border: 1px solid rgba(0,0,0,0.06);
}
.form-title {
  text-align: center;
  font-size: 1.9rem;
  font-weight: 700;
  margin: 10px 0 14px 0;
  color: #222;
}
.info-box {
  background: linear-gradient(135deg, #ffffff, #f2f4ff);
  padding: 10px 12px;
  border-radius: 10px;
  border: 1px solid #e6e9f0;
  font-size: 14px;
  font-weight: 600;
  margin-bottom: 10px;
  text-align: center;
  color: #111;
  box-shadow: 0 2px 8px rgba(0,0,0,0.08);
}
.input-card {
  background: #ffffff;
  padding: 16px;
  border-radius: 12px;
  box-shadow: 0 4px 18px rgba(0,0,0,0.08);
  border: 1px solid rgba(0,0,0,0.05);
}
footer { visibility: hidden; }
.custom-footer {
  text-align: center;
  color: #111;
  padding: 8px 0 12px 0;
  font-size: 13px;
}
.custom-footer a { color: #111; text-decoration: none; }
.result-tag {
  display: inline-block;
  padding: 6px 10px;
  border-radius: 8px;
  font-weight: 700;
  border: 1px solid rgba(0,0,0,0.08);
}
.hr-soft {
  height: 1px;
  background: linear-gradient(90deg, rgba(0,0,0,0), rgba(0,0,0,0.12), rgba(0,0,0,0));
  border: none;
  margin: 12px 0;
}
</style>
""", unsafe_allow_html=True)

# ===============================
# Sidebar
# ===============================
with st.sidebar:
    st.markdown("""
    <div class="sidebar-card">
        <h3 style="margin:0 0 6px 0;">Ume Habiba</h3>
        <div style="font-size:14px;">University of Narowal</div>
        <div style="font-size:14px; margin-top:6px;">📧 <a href="mailto:ar2838294@gmail.com">ar2838294@gmail.com</a></div>
        <hr class="hr-soft"/>
        <div style="font-size:13px; line-height:1.4;">
            Use the form on the right to enter patient details and get a risk prediction.
        </div>
    </div>
    """, unsafe_allow_html=True)

# ===============================
# Main Title
# ===============================
st.markdown('<div class="form-title">Diabetes Risk Prediction</div>', unsafe_allow_html=True)

# ===============================
# Input Form
# ===============================
st.markdown('<div class="info-box">Enter your details</div>', unsafe_allow_html=True)
st.markdown('<div class="input-card">', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    glucose = st.number_input("Glucose Level", min_value=0, max_value=300, value=100, step=1)
    insulin = st.number_input("Insulin", min_value=0, max_value=900, value=80, step=1)
    age = st.number_input("Age", min_value=1, max_value=120, value=30, step=1)
with col2:
    blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=80, step=1)
    bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0, step=0.1)
predict_clicked = st.button("Predict", type="primary")
st.markdown('</div>', unsafe_allow_html=True)

# ===============================
# Predict & Display
# ===============================
if predict_clicked:
    try:
        X = np.array([[glucose, blood_pressure, insulin, bmi, age]])
        if scaler is not None:
            X = scaler.transform(X)

        pred = model.predict(X)[0]
        proba = model.predict_proba(X)[0][1] if hasattr(model, "predict_proba") else None

        # Determine category
        if proba is not None:
            if proba < 0.40:
                category = "Not Diabetic"; slider_val = 0.0; slider_color = "green"
            elif proba < 0.60:
                category = "Pre-Diabetic"; slider_val = 0.5; slider_color = "gold"
            else:
                category = "Diabetic"; slider_val = 1.0; slider_color = "red"
        else:
            category = "Diabetic" if pred == 1 else "Not Diabetic"
            slider_val = 1.0 if pred == 1 else 0.0
            slider_color = "red" if pred == 1 else "green"

        # Create side-by-side layout
        left_block, right_block = st.columns([2, 1])

        with left_block:
            # Display results & instructions
            if category == "Diabetic":
                st.error(f"⚠️ Patient is **Diabetic**" + (f" ({proba:.2%} risk)" if proba is not None else ""))
                st.markdown("""
                **Instructions for Diabetic Patients:**
                - Monitor blood glucose regularly.
                - Follow a balanced, low-sugar diet.
                - Take prescribed medication/insulin as directed.
                - Engage in 30 minutes of moderate exercise daily.
                - Schedule regular check-ups with your healthcare provider.
                """)
            elif category == "Pre-Diabetic":
                st.warning(f"⚠️ Patient is **Pre-Diabetic**" + (f" ({proba:.2%} risk)" if proba is not None else ""))
                st.markdown("""
                **Instructions for Pre-Diabetic Patients:**
                - Reduce refined carbohydrates and added sugars.
                - Increase daily physical activity (at least 150 minutes/week).
                - Maintain a healthy weight.
                - Monitor blood glucose and blood pressure regularly.
                - Get a health check-up every 6–12 months.
                """)
            else:
                st.success(f"✅ Patient is **Not Diabetic**" + (f" ({proba:.2%} risk)" if proba is not None else ""))
                st.markdown("""
                **Healthy Living Tips:**
                - Maintain balanced diet, daily exercise, hydration, and routine screening.
                """)

            # Slider visual
            st.markdown(f"""
            <div style="margin-top:14px; text-align:center;">
                <div style="width:100%; background:linear-gradient(to right, green 33%, gold 66%, red 100%);
                            height:16px; border-radius:10px; position:relative; border:1px solid rgba(0,0,0,0.15);">
                    <div style="position:absolute; left:{slider_val*100}%; top:-7px; transform:translateX(-50%);
                                background:{slider_color}; width:16px; height:30px; border-radius:6px;
                                border:2px solid rgba(0,0,0,0.35);"></div>
                </div>
                <div style="display:flex; justify-content:space-between; font-size:12px; margin-top:6px; color:#111;">
                    <span>Not Diabetic</span><span>Pre-Diabetic</span><span>Diabetic</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with right_block:
            fig, ax = plt.subplots(figsize=(3, 2))
            bars = ax.bar(
                ["Glucose", "BloodPressure", "Insulin", "BMI", "Age"],
                [glucose, blood_pressure, insulin, bmi, age],
                color=["#7db3ff", "#a5d8ff", "#b8f2e6", "#f6c0ff", "#ffd6a5"],
                edgecolor="#333", linewidth=0.6
            )
            ax.set_ylabel("Value", fontsize=8, color="#111")
            ax.set_title("Patient Health Metrics", fontsize=9, color="#111", pad=4)
            ax.set_facecolor("#f7f8fb")
            ax.grid(axis='y', linestyle="--", alpha=0.5, linewidth=0.6)

            for bar in bars:
                h = bar.get_height()
                ax.annotate(f'{h:.1f}' if isinstance(h, float) else f'{int(h)}',
                            xy=(bar.get_x() + bar.get_width() / 2, h),
                            xytext=(0, 2),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=6, color="#111")
            plt.xticks(fontsize=8, rotation=20)
            plt.yticks(fontsize=8)
            ax.set_ylim(0, max(glucose, blood_pressure, insulin, bmi, age) * 1.15)

            st.pyplot(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error during prediction: {e}")

# ===============================
# Footer
# ===============================
st.markdown('<div class="custom-footer">📧 <a href="mailto:ar2838294@gmail.com">ar2838294@gmail.com</a></div>', unsafe_allow_html=True)
