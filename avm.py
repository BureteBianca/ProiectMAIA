import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from scipy.stats import kurtosis, skew
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, Normalizer, RobustScaler, \
    QuantileTransformer
import warnings
import pymongo

warnings.filterwarnings('ignore')

# Configurare pagină
st.set_page_config(
    page_title="Proiect",
    page_icon="     ",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizat
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
    }
    .method-box {
        background-color: #f0f8ff;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #3498db;
        margin: 1rem 0;
    }
    .why-box {
        background-color: #fff9e6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 5px solid #f39c12;
        margin: 1rem 0;
    }
    .theory-box {
        background-color: #f0f0f0;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .code-explanation {
        background-color: #e8f5e9;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 8px;
        border-left: 5px solid #28a745;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def load_data_from_csv(uploaded_file):
    """Încarcă datele dintr-un fișier CSV încărcat în Streamlit."""
    df = pd.read_csv(uploaded_file)
    return df
def sidebar_navigation():
    st.sidebar.markdown("#  Seminar AVM")
    st.sidebar.markdown("### Navighează:")

    sections = [
        " Încărcare Date"
    ]

    selected = st.sidebar.radio("Selectează Modulul:", sections)

    st.sidebar.markdown("---")

    st.sidebar.markdown("---")
    st.sidebar.markdown("###   Etapele Analizei")
    st.sidebar.markdown("""
    1. **Organizare date** - MongoDB, SQL
    2. **Prelucrare date** - Curățare, standardizare
    3. **Analiză date** - Descriptivă, diagnostic, predictivă
    """)

    return selected

# Date
def show_data_connection():
    st.markdown('<h1 class="main-header"> Încărcare Date</h1>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Alternativă: Încarcă datele dintr-un fișier CSV</div>', unsafe_allow_html=True)

    with st.expander(" Încarcă fișier CSV (fără MongoDB)"):
        uploaded_csv = st.file_uploader("Alege fișierul CSV:", type=["csv"])

        if uploaded_csv is not None:
            try:
                df_csv = load_data_from_csv(uploaded_csv)
                st.session_state['df'] = df_csv
                st.session_state['collection_name'] = "csv_upload"

                st.success(f"Date încărcate cu succes din CSV! ({len(df_csv):,} rânduri, {len(df_csv.columns)} coloane)")

                st.dataframe(df_csv.head(10), use_container_width=True)

            except Exception as e:
                st.error(f"Eroare la citirea CSV-ului: {e}")

    # Display data if loaded
    if 'df' in st.session_state:
        df = st.session_state['df']
        collection = st.session_state.get('collection_name', 'unknown')

        st.markdown('<div class="sub-header">Pasul 3: Explorarea Datelor Încărcate</div>', unsafe_allow_html=True)

        # Data overview
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Rânduri", f"{len(df):,}")

        with col2:
            st.metric("Total Coloane", len(df.columns))

        with col3:
            memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
            st.metric(" Memorie", f"{memory_mb:.2f} MB")

        with col4:
            missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100)
            st.metric(" Valori Lipsă", f"{missing_pct:.1f}%")

        # Tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs([" Preview", " Info", " Statistici", " Vizualizare"])

        with tab1:
            st.markdown("### Primele Rânduri")
            n_rows = st.slider("Număr rânduri de afișat:", 5, 50, 10, key="preview_rows")
            st.dataframe(df.head(n_rows), use_container_width=True)

            with st.expander("Ultimele Rânduri"):
                st.dataframe(df.tail(n_rows), use_container_width=True)

        with tab2:
            st.markdown("### Informații Dataset")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Tipuri de Date:**")
                dtype_df = pd.DataFrame({
                    'Coloană': df.columns,
                    'Tip': df.dtypes.astype(str),
                    'Non-Null': df.count().values,
                    'Null': df.isnull().sum().values
                })
                st.dataframe(dtype_df, use_container_width=True)

            with col2:
                st.markdown("**Distribuția Tipurilor:**")
                type_counts = df.dtypes.astype(str).value_counts()
                fig = px.pie(
                    values=type_counts.values,
                    names=type_counts.index,
                    title="Tipuri de Date"
                )
                st.plotly_chart(fig, use_container_width=True)

        with tab3:
            st.markdown("### Statistici Descriptive")

            # Numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                st.markdown("**Coloane Numerice:**")
                st.dataframe(df[numeric_cols].describe(), use_container_width=True)

            # Categorical columns
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            if categorical_cols:
                st.markdown("**Coloane Categorice:**")
                cat_summary = pd.DataFrame({
                    col: [
                        df[col].nunique(),
                        df[col].mode()[0] if len(df[col].mode()) > 0 else None,
                        df[col].value_counts().iloc[0] if len(df[col]) > 0 else 0,
                        f"{(df[col].value_counts().iloc[0] / len(df) * 100):.1f}%" if len(df[col]) > 0 else "0%"
                    ] for col in categorical_cols
                }, index=['Valori Unice', 'Cel Mai Comun', 'Frecvență', 'Procent']).T
                st.dataframe(cat_summary, use_container_width=True)

        with tab4:
            st.markdown("### Vizualizare Valori Lipsă")

            missing = df.isnull().sum()
            missing_pct = (missing / len(df)) * 100
            missing_df = pd.DataFrame({
                'Coloană': missing.index,
                'Număr Lipsă': missing.values,
                'Procent': missing_pct.values
            }).sort_values('Număr Lipsă', ascending=False)

            cols_with_missing = missing_df[missing_df['Număr Lipsă'] > 0]

            if len(cols_with_missing) > 0:
                fig = px.bar(
                    cols_with_missing,
                    x='Coloană',
                    y='Procent',
                    title='Procentul Valorilor Lipsă pe Coloană',
                    text='Număr Lipsă'
                )
                fig.update_traces(textposition='outside')
                st.plotly_chart(fig, use_container_width=True)

                st.dataframe(cols_with_missing, use_container_width=True)
            else:
                st.success(" Nu există valori lipsă în dataset!")

            # Heatmap for missing values
            if len(cols_with_missing) > 0:
                st.markdown("### Heatmap Valori Lipsă (primele 50 rânduri)")
                colours = ['#ffff00', '#000099']  # yellow = missing, blue = present
                fig, ax = plt.subplots(figsize=(12, 8))
                sns.heatmap(df.head(50).isnull(), cmap=sns.color_palette(colours),
                            cbar=False, yticklabels=False, ax=ax)
                ax.set_title("Galben = Lipsă, Albastru = Prezent")
                st.pyplot(fig)

        st.markdown('<div class="success-box">', unsafe_allow_html=True)
        st.markdown(f"""
        ###  Date Încărcate cu Succes!

        Ai încărcat **{len(df):,} rânduri** și **{len(df.columns)} coloane** din colecția `{collection}`.

        Poți continua acum cu metodele de prelucrare a datelor în secțiunile următoare! 
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.info(" Configurează conexiunea și apasă butonul 'Încarcă Date' pentru a începe!")


