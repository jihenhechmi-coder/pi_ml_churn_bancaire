import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Churn Bancaire | Pilotage Client", page_icon="🏦", layout="wide")

# ======================================================================
# STYLE — Identité visuelle bancaire (navy / or)
# ======================================================================
NAVY = "#0B2545"
NAVY_2 = "#13315C"
GOLD = "#B8860B"
GOLD_LIGHT = "#D9A93C"
BG = "#F4F6F9"
GREEN = "#1B7A43"
RED = "#B3261E"
GREY = "#5C6B7A"

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Libre+Franklin:wght@400;500;600;700;800&family=IBM+Plex+Mono:wght@500&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Libre Franklin', -apple-system, sans-serif;
    }}

    .stApp {{
        background-color: {BG};
    }}

    /* ---- Bannière d'en-tête ---- */
    .bank-header {{
        background: linear-gradient(120deg, {NAVY} 0%, {NAVY_2} 100%);
        padding: 28px 36px;
        border-radius: 10px;
        margin-bottom: 28px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        border-bottom: 3px solid {GOLD};
    }}
    .bank-header h1 {{
        color: #FFFFFF;
        font-size: 26px;
        font-weight: 800;
        letter-spacing: 0.3px;
        margin: 0;
    }}
    .bank-header p {{
        color: #C9D4E3;
        font-size: 13.5px;
        margin: 4px 0 0 0;
        font-weight: 500;
    }}
    .bank-badge {{
        background: rgba(184, 134, 11, 0.18);
        border: 1px solid {GOLD};
        color: {GOLD_LIGHT};
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 0.4px;
        white-space: nowrap;
    }}

    /* ---- Cartes KPI ---- */
    .kpi-card {{
        background: #FFFFFF;
        border: 1px solid #E3E8EF;
        border-left: 4px solid {NAVY};
        border-radius: 8px;
        padding: 18px 20px;
        box-shadow: 0 1px 3px rgba(11,37,69,0.06);
        height: 100%;
    }}
    .kpi-card.alert {{ border-left-color: {RED}; }}
    .kpi-card.ok {{ border-left-color: {GREEN}; }}
    .kpi-card.gold {{ border-left-color: {GOLD}; }}
    .kpi-label {{
        color: {GREY};
        font-size: 12.5px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        margin-bottom: 6px;
    }}
    .kpi-value {{
        color: {NAVY};
        font-size: 30px;
        font-weight: 800;
        font-family: 'IBM Plex Mono', monospace;
        line-height: 1.1;
    }}
    .kpi-sub {{
        color: {GREY};
        font-size: 12.5px;
        margin-top: 6px;
        font-weight: 500;
    }}

    /* ---- Sidebar ---- */
    section[data-testid="stSidebar"] {{
        background-color: {NAVY};
    }}
    section[data-testid="stSidebar"] * {{
        color: #E8EDF4 !important;
    }}
    section[data-testid="stSidebar"] .stRadio label {{
        font-weight: 500;
    }}
    section[data-testid="stSidebar"] hr {{
        border-color: rgba(255,255,255,0.15);
    }}

    /* ---- Titres de section ---- */
    h1, h2, h3 {{
        color: {NAVY} !important;
        font-weight: 700 !important;
    }}
    .section-eyebrow {{
        color: {GOLD};
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 1.2px;
        text-transform: uppercase;
        margin-bottom: 2px;
    }}

    /* ---- Formulaire ---- */
    div[data-testid="stForm"] {{
        background: #FFFFFF;
        border: 1px solid #E3E8EF;
        border-radius: 10px;
        padding: 24px 26px;
    }}
    .stButton button, div[data-testid="stFormSubmitButton"] button {{
        background-color: {NAVY} !important;
        color: #FFFFFF !important;
        border: none !important;
        font-weight: 600 !important;
        border-radius: 6px !important;
        padding: 0.5rem 1.6rem !important;
    }}
    .stButton button:hover, div[data-testid="stFormSubmitButton"] button:hover {{
        background-color: {GOLD} !important;
        color: {NAVY} !important;
    }}

    /* ---- Dataframe ---- */
    .stDataFrame {{
        border: 1px solid #E3E8EF;
        border-radius: 8px;
    }}

    footer, #MainMenu {{ visibility: hidden; }}
</style>
""", unsafe_allow_html=True)


def kpi_card(label, value, sub="", style="navy"):
    st.markdown(f"""
    <div class="kpi-card {style}">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-sub">{sub}</div>
    </div>
    """, unsafe_allow_html=True)


def header(title, eyebrow):
    st.markdown(f"""<div class="section-eyebrow">{eyebrow}</div>""", unsafe_allow_html=True)
    st.markdown(f"## {title}")


PLOTLY_LAYOUT = dict(
    font=dict(family="Libre Franklin, sans-serif", color=NAVY),
    colorway=[NAVY, GOLD, "#5C7A99", "#C9A227", RED, GREEN],
    plot_bgcolor="white",
    paper_bgcolor="white",
    title_font=dict(size=15, color=NAVY),
    margin=dict(t=50, l=10, r=10, b=10),
)

# ======================================================================
# Chargement des artefacts et des données
# ======================================================================
@st.cache_resource
def load_model():
    model = joblib.load("model_churn.pkl")
    features = joblib.load("features_churn.pkl")
    return model, features

@st.cache_data(ttl=3600)
def load_data():
    return pd.read_csv("fact_client_ml.csv")

model, FEATURES = load_model()
df = load_data()

# ======================================================================
# En-tête + Sidebar
# ======================================================================
st.markdown(f"""
<div class="bank-header">
    <div>
        <h1>🏦 Plateforme de Pilotage du Churn Client</h1>
        <p>Direction Risques &amp; Rétention — Prédiction et suivi des départs clients</p>
    </div>
    <div class="bank-badge">MODÈLE XGBOOST · PRODUCTION</div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("### Navigation")
page = st.sidebar.radio("", ["📊 Tableau de bord", "🔮 Prédiction individuelle", "⚠️ Comptes à risque"],
                         label_visibility="collapsed")
st.sidebar.markdown("---")
SEUIL = st.sidebar.slider("Seuil de risque (probabilité)", 0.30, 0.90, 0.50, 0.05)
st.sidebar.markdown("---")
st.sidebar.markdown("**Performance du modèle**")
st.sidebar.markdown(
    "<div style='font-size:13px; line-height:1.9'>"
    "Recall &nbsp;<b>0.842</b><br>"
    "F1-score &nbsp;<b>0.830</b><br>"
    "Accuracy &nbsp;<b>0.885</b>"
    "</div>", unsafe_allow_html=True
)

# ======================================================================
# PAGE 1 : Tableau de bord
# ======================================================================
if page == "📊 Tableau de bord":
    header("Vue d'ensemble du portefeuille", "TABLEAU DE BORD")

    actifs = df[df["CHURN"] == 0] if "CHURN" in df.columns else df
    a_risque = actifs[actifs["churn_proba"] >= SEUIL]

    taux = len(a_risque) / len(actifs) * 100 if len(actifs) else 0
    pnb_danger = a_risque["pnb_proxy"].sum()
    pnb_total = actifs["pnb_proxy"].sum()
    part_pnb = pnb_danger / pnb_total * 100 if pnb_total else 0

    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi_card("Portefeuille actif", f"{len(actifs):,}".replace(",", " "), "clients suivis", "navy")
    with c2: kpi_card("Taux de churn prédit", f"{taux:.1f} %", f"au seuil de {SEUIL:.0%}", "alert")
    with c3: kpi_card("Clients à risque", f"{len(a_risque):,}".replace(",", " "), "à contacter en priorité", "alert")
    with c4: kpi_card("PNB en danger", f"{pnb_danger:,.0f} TND".replace(",", " "), f"{part_pnb:.1f} % du PNB total", "gold")

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(actifs, x="churn_proba", nbins=40,
                           title="Distribution des probabilités de churn")
        fig.update_traces(marker_color=NAVY)
        fig.add_vline(x=SEUIL, line_dash="dash", line_color=RED,
                      annotation_text="Seuil", annotation_font_color=RED)
        fig.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        if "CLUSTER" in actifs.columns:
            agg = actifs.groupby("CLUSTER").agg(
                taux=("churn_proba", lambda s: (s >= SEUIL).mean() * 100),
                pnb_danger=("pnb_proxy", lambda s: s[actifs.loc[s.index, "churn_proba"] >= SEUIL].sum())
            ).reset_index()
            fig2 = px.bar(agg, x="CLUSTER", y="taux", color="pnb_danger",
                          title="Taux de churn par segment (cluster)",
                          labels={"taux": "% clients à risque", "pnb_danger": "PNB en danger"},
                          color_continuous_scale=[NAVY, GOLD])
            fig2.update_layout(**PLOTLY_LAYOUT)
            st.plotly_chart(fig2, use_container_width=True)

    fig3 = px.scatter(actifs.sample(min(3000, len(actifs))),
                      x="anciennete_client", y="churn_proba", color="pnb_proxy",
                      opacity=0.55, title="Ancienneté vs probabilité de churn",
                      labels={"anciennete_client": "Ancienneté (années)"},
                      color_continuous_scale=[NAVY, GOLD])
    fig3.update_layout(**PLOTLY_LAYOUT)
    st.plotly_chart(fig3, use_container_width=True)

# ======================================================================
# PAGE 2 : Prédiction individuelle
# ======================================================================
elif page == "🔮 Prédiction individuelle":
    header("Estimation du risque de départ", "SIMULATEUR CLIENT")
    st.caption("Renseignez le profil du client pour estimer sa probabilité de départ.")

    with st.form("form_client"):
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input("Âge du client", 18, 100, 40)
            anciennete = st.number_input("Ancienneté (années)", 0, 60, 5)
            salary = st.number_input("Salaire imputé (TND)", 0.0, 50000.0, 1500.0, step=100.0)
            pnb = st.number_input("PNB proxy (TND)", 0.0, 100000.0, 500.0, step=50.0)
        with c2:
            avancement = st.slider("Avancement crédit", 0.0, 1.0, 0.0)
            nb_credits = st.number_input("Nb crédits", 0, 10, 0)
            nb_depots = st.number_input("Nb dépôts", 0, 10, 0)
            segment = st.selectbox("Segment client", ["Retail", "Elite"])
        with c3:
            risque_eleve = st.checkbox("Risque élevé")
            a_depot_terme = st.checkbox("A un dépôt à terme")
            non_resident = st.checkbox("Non résident")
            flag_impaye = st.checkbox("Impayé")
            a_credit_corp = st.checkbox("Crédit corporate")

        submitted = st.form_submit_button("Calculer le score de risque", type="primary")

    if submitted:
        row = {
            "age_client": age, "anciennete_client": anciennete,
            "salary_impute": salary, "risque_eleve": int(risque_eleve),
            "pnb_proxy": pnb, "avancement_credit": avancement,
            "a_depot_terme": int(a_depot_terme),
            "nb_credits": nb_credits, "nb_depots": nb_depots,
            "non_resident": int(non_resident),
            "flag_impaye": int(flag_impaye), "a_credit_corporate": int(a_credit_corp),
            "SEG_Retail": 1 if segment == "Retail" else 0,
        }
        X = pd.DataFrame([row]).reindex(columns=FEATURES, fill_value=0)
        proba = float(model.predict_proba(X)[0, 1])

        st.markdown("<br>", unsafe_allow_html=True)
        col_a, col_b = st.columns([1, 2])
        with col_a:
            kpi_card("Probabilité de churn", f"{proba*100:.1f} %", "score du modèle",
                     "alert" if proba >= SEUIL else "ok")
        with col_b:
            if proba >= SEUIL:
                st.error(f"⚠️ **Client à risque** (seuil {SEUIL:.0%}). Action de rétention recommandée : "
                        "contact conseiller, offre de fidélisation, révision tarifaire.")
            else:
                st.success("✅ **Client stable** au seuil actuel. Aucune action requise.")
        st.progress(min(proba, 1.0))

# ======================================================================
# PAGE 3 : Comptes à risque
# ======================================================================
else:
    header("Portefeuille des clients à retenir", "ACTION COMMERCIALE")
    st.caption("Clients actifs (non churned) triés par probabilité de départ décroissante — "
               "priorisez le contact par le conseiller.")

    actifs = df[df["CHURN"] == 0] if "CHURN" in df.columns else df
    risk = actifs[actifs["churn_proba"] >= SEUIL].sort_values("churn_proba", ascending=False)

    c1, c2 = st.columns(2)
    with c1:
        if "CLUSTER" in df.columns:
            clusters = st.multiselect("Filtrer par cluster", sorted(df["CLUSTER"].unique()))
            if clusters:
                risk = risk[risk["CLUSTER"].isin(clusters)]
    with c2:
        pnb_min = st.number_input("PNB minimum (TND)", 0.0, value=0.0, step=100.0)
        risk = risk[risk["pnb_proxy"] >= pnb_min]

    c1, c2 = st.columns(2)
    with c1: kpi_card("Clients identifiés", f"{len(risk):,}".replace(",", " "), f"seuil {SEUIL:.0%}", "alert")
    with c2: kpi_card("PNB cumulé en jeu", f"{risk['pnb_proxy'].sum():,.0f} TND".replace(",", " "), "à sécuriser", "gold")

    st.markdown("<br>", unsafe_allow_html=True)

    cols = [c for c in ["CUSTOMER_NO", "churn_proba", "pnb_proxy", "SOLDE_DAV",
                        "anciennete_client", "age_client", "CLUSTER",
                        "compte_dormant", "flag_impaye"] if c in risk.columns]
    st.dataframe(
        risk[cols].style.background_gradient(subset=["churn_proba"], cmap="YlOrBr")
                        .format({"churn_proba": "{:.1%}", "pnb_proxy": "{:,.0f}",
                                 "SOLDE_DAV": "{:,.0f}"}),
        use_container_width=True, height=520)

    st.download_button("📥 Exporter la liste (CSV)",
                       risk[cols].to_csv(index=False).encode("utf-8"),
                       "comptes_a_risque.csv", "text/csv")
