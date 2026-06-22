# 🎓 AI-Powered Student Dropout Prediction System
### Bharat Academix CodeQuest Hackathon 2026

An ML-powered early warning dashboard that predicts student dropout risk
for government schools across India — before it happens.

---

## 👥 Team
| Role | Name |
|------|------|
| Team Leader | Sudham Hasini |
| Team Member | Pillalamarri Harshitha |
| Team Member | Pillalamarri Swastika |

---

## 🚨 Problem
Every year, 1.5 crore students drop out of government schools in India,
mostly between Classes 6–10. Teachers managing 40–60 students have no
early warning system — by the time a dropout is noticed, it's too late.

---

## ✅ Our Solution
A Machine Learning system that:
- Assigns every student a **dropout risk score** — Low / Medium / High
- Explains **why** a student is at risk (using SHAP)
- Recommends **specific interventions** per student
- Displays everything on a **teacher-friendly Streamlit dashboard**

---

## 🛠️ Tech Stack
| Layer | Technology |
|-------|-----------|
| ML Model | XGBoost Classifier |
| Explainability | SHAP (SHapley Values) |
| Dashboard | Streamlit |
| Data Processing | Pandas, NumPy |
| Visualisation | Matplotlib |
| Dataset | Synthetic (based on UDISE+/ASER patterns) |

---

## 📊 Model Performance
| Metric | Score |
|--------|-------|
| Overall Accuracy | 93% |
| Dropout Precision | 81% |
| Not Dropout Precision | 94% |
| Training Samples | 2,000 students |

---

## 🚀 How to Run

### 1. Clone the repository