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
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score
)


warnings.filterwarnings('ignore')

# Configurare pagină
st.set_page_config(
    page_title="Proiect",
    page_icon="     ",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    body, .stApp {
        background-color: #001F54;  
    }
    .main-header, .sub-header, .method-box, .why-box, .theory-box, .code-explanation, .success-box, 
    .stMarkdown, .stText {
        color: #DA70D6;  
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def load_data(uploaded_file):
    """Încarcă date din CSV sau Excel"""
    file_name = uploaded_file.name.lower()

    if file_name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)

    elif file_name.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(uploaded_file)

    else:
        raise ValueError("Format fișier neacceptat!")

    return df
def sidebar_navigation():
    st.sidebar.markdown("#  Proiect MAIA + ML")
    st.sidebar.markdown("### Navighează:")

    sections = [
        " Incarcare date",
        " Filtrare date",
        " Analiza valori lipsa",
        " Detectarea valorilor anormale",
        " Prelucrarea sirurilor de caractere",
        " Standardizare si normalizare",
        " Statistici descriptive",
        " Reprezentari grafice",
        " Problem Setup - ML",
        " Preprocesare & Pipeline - ML",
        " Train/Test Split - ML",
        " Models - ML",
        " Evaluare si comparare - ML"
    ]

    selected = st.sidebar.radio("Selectează Modulul:", sections)

    st.sidebar.markdown("---")

    return selected

# Date
def show_data_connection():
    st.markdown('<h1 class="main-header"> Încărcare Date</h1>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Încarcă datele dintr-un fișier CSV/Excel</div>', unsafe_allow_html=True)

    with st.expander(" Încarcă fișier CSV sau Excel"):
        uploaded_file = st.file_uploader(
            "Alege fișierul (CSV / XLSX / XLS):",
            type=["csv", "xlsx", "xls"]
        )

        if uploaded_file is not None:
            try:
                df = load_data(uploaded_file)
                st.session_state['df'] = df
                st.session_state['collection_name'] = uploaded_file.name

                st.success(
                    f"Date încărcate cu succes! "
                    f"({len(df):,} rânduri, {len(df.columns)} coloane)"
                )

                st.dataframe(df.head(10), use_container_width=True)

            except Exception as e:
                st.error(f"Eroare la citirea fișierului: {e}")

    # Display data if loaded
    if 'df' in st.session_state:
        df = st.session_state['df']
        collection = st.session_state.get('collection_name', 'unknown')

        st.markdown('<div class="sub-header">Explorarea Datelor Încărcate</div>', unsafe_allow_html=True)

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

#Filtrare

def show_data_filtering():
    st.markdown('<h1 class="main-header"> Filtrare Date</h1>', unsafe_allow_html=True)

    df = st.session_state['df'].copy()

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()

    st.markdown('<div class="sub-header">Filtrare Coloane Numerice</div>', unsafe_allow_html=True)

    filtered_df = df.copy()

    # -------------------- FILTRARE NUMERICĂ --------------------
    if numeric_cols:
        num_cols_selected = st.multiselect(
            "Selectează coloanele numerice pentru filtrare:",
            numeric_cols,
            default=numeric_cols[:1]
        )

        for col in num_cols_selected:
            min_val = float(df[col].min())
            max_val = float(df[col].max())

            selected_range = st.slider(
                f"{col} (interval):",
                min_value=min_val,
                max_value=max_val,
                value=(min_val, max_val)
            )

            filtered_df = filtered_df[
                (filtered_df[col] >= selected_range[0]) &
                (filtered_df[col] <= selected_range[1])
            ]

    else:
        st.info("Nu există coloane numerice pentru filtrare.")

    st.markdown('<div class="sub-header">Filtrare Coloane Categorice</div>', unsafe_allow_html=True)

    # -------------------- FILTRARE CATEGORICĂ --------------------
    if cat_cols:
        cat_cols_selected = st.multiselect(
            "Selectează coloanele categorice pentru filtrare:",
            cat_cols
        )

        for col in cat_cols_selected:
            unique_values = df[col].dropna().unique().tolist()

            selected_values = st.multiselect(
                f"Valori permise pentru {col}:",
                unique_values,
                default=unique_values
            )

            filtered_df = filtered_df[filtered_df[col].isin(selected_values)]

    else:
        st.info("Nu există coloane categorice pentru filtrare.")

    # -------------------- REZULTATE --------------------
    st.markdown('<div class="sub-header">Rezultat Filtrare</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Rânduri inițiale", len(df))

    with col2:
        st.metric("Rânduri după filtrare", len(filtered_df))

    # Procent păstrat
    pct_remaining = (len(filtered_df) / len(df) * 100) if len(df) > 0 else 0
    st.metric("Procent date păstrate", f"{pct_remaining:.2f}%")

    st.markdown("### DataFrame Filtrat")
    st.dataframe(filtered_df, use_container_width=True)

    # Salvare în session_state
    st.session_state['df_filtered_final'] = filtered_df


# Valori lipsa
def show_missing_values_analysis():
    st.markdown('<h1 class="main-header"> Analiza Valorilor Lipsă</h1>', unsafe_allow_html=True)

    df = st.session_state['df'].copy()

    # -------------------- IDENTIFICARE VALORI LIPSĂ --------------------
    missing_count = df.isnull().sum()
    missing_percent = (missing_count / len(df)) * 100

    missing_df = pd.DataFrame({
        'Coloană': missing_count.index,
        'Valori Lipsă': missing_count.values,
        'Procent (%)': missing_percent.values
    }).sort_values('Valori Lipsă', ascending=False)

    # Coloane cu valori lipsă
    cols_with_missing = missing_df[missing_df['Valori Lipsă'] > 0]

    st.markdown("### Coloane cu valori lipsă")

    if cols_with_missing.empty:
        st.success("✅ Nu există valori lipsă în dataset!")
        return

    st.dataframe(cols_with_missing, use_container_width=True)

    # -------------------- GRAFIC BAR --------------------
    st.markdown("### Vizualizare – Procent valori lipsă")

    fig = px.bar(
        cols_with_missing,
        x='Coloană',
        y='Procent (%)',
        text='Valori Lipsă',
        title='Procentul valorilor lipsă pe coloană'
    )

    fig.update_traces(textposition='outside')
    fig.update_layout(
        xaxis_title="Coloană",
        yaxis_title="Procent valori lipsă (%)"
    )

    st.plotly_chart(fig, use_container_width=True)

    # -------------------- HEATMAP (OPȚIONAL, BONUS) --------------------
    st.markdown("### Heatmap valori lipsă (primele 50 rânduri)")

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(
        df.head(50).isnull(),
        cbar=False,
        yticklabels=False,
        cmap=['#001F54', '#DA70D6']
    )
    ax.set_title("Mov = Lipsă | Albastru = Prezent")
    st.pyplot(fig)

def show_outlier_detection():
    st.markdown('<h1 class="main-header"> Detectarea Valorilor Anormale (Outlieri)</h1>', unsafe_allow_html=True)

    df = st.session_state['df'].copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if not numeric_cols:
        st.error("Nu există coloane numerice în dataset!")
        return

    st.markdown('<div class="sub-header">Analiză cu Histogramă</div>', unsafe_allow_html=True)

    col_for_hist = st.selectbox("Selectează coloana pentru histogramă:", numeric_cols, key="hist_col")

    col1, col2 = st.columns([2, 1])

    with col1:
        fig = px.histogram(
            df,
            x=col_for_hist,
            nbins=50,
            title=f'Histogramă: {col_for_hist}',
            marginal='box'
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### Statistici")
        stats_df = pd.DataFrame({
            'Metrică': ['Minim', 'Q1 (25%)', 'Mediană', 'Q3 (75%)', 'Maxim', 'Media', 'Std Dev'],
            'Valoare': [
                df[col_for_hist].min(),
                df[col_for_hist].quantile(0.25),
                df[col_for_hist].median(),
                df[col_for_hist].quantile(0.75),
                df[col_for_hist].max(),
                df[col_for_hist].mean(),
                df[col_for_hist].std()
            ]
        })
        st.dataframe(stats_df, use_container_width=True)

    st.markdown('<div class="sub-header">Box Plot</div>', unsafe_allow_html=True)

    col_for_box = st.selectbox("Selectează coloana pentru box plot:", numeric_cols, key="box_col")

    # Calculate IQR and outliers
    Q1 = df[col_for_box].quantile(0.25)
    Q3 = df[col_for_box].quantile(0.75)
    IQR = Q3 - Q1
    lower_fence = Q1 - 1.5 * IQR
    upper_fence = Q3 + 1.5 * IQR

    outliers = df[(df[col_for_box] < lower_fence) | (df[col_for_box] > upper_fence)]
    n_outliers = len(outliers)
    pct_outliers = (n_outliers / len(df) * 100) if len(df) > 0 else 0

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(" Total Valori", len(df))

    with col2:
        st.metric("Outlieri Găsiți", n_outliers)

    with col3:
        st.metric(" Procent Outlieri", f"{pct_outliers:.2f}%")

    # Box plot
    fig = px.box(
        df,
        y=col_for_box,
        points='outliers',
        title=f'Box Plot: {col_for_box}'
    )
    fig.add_hline(y=lower_fence, line_dash="dash", line_color="red", annotation_text="Lower Fence")
    fig.add_hline(y=upper_fence, line_dash="dash", line_color="red", annotation_text="Upper Fence")
    st.plotly_chart(fig, use_container_width=True)

    if n_outliers > 0:
        with st.expander(" Vezi Outlierii"):
            st.dataframe(outliers[[col_for_box]].describe(), use_container_width=True)
            st.dataframe(outliers.head(20), use_container_width=True)

    st.markdown('<div class="sub-header">Metoda 3: Detectare cu Quantile</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        q_low_pct = st.slider("Quantila inferioară (%):", 0.0, 10.0, 1.0, 0.5)

    with col2:
        q_high_pct = st.slider("Quantila superioară (%):", 90.0, 100.0, 99.0, 0.5)

    col_for_quantile = st.selectbox("Selectează coloana:", numeric_cols, key="quantile_col")

    q_low = df[col_for_quantile].quantile(q_low_pct / 100)
    q_high = df[col_for_quantile].quantile(q_high_pct / 100)

    df_filtered = df[(df[col_for_quantile] >= q_low) & (df[col_for_quantile] <= q_high)]
    n_removed = len(df) - len(df_filtered)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Înainte Filtrare")
        fig1 = px.box(df, y=col_for_quantile, title="Original")
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.markdown("### După Filtrare")
        fig2 = px.box(df_filtered, y=col_for_quantile, title="Filtrat")
        st.plotly_chart(fig2, use_container_width=True)

    if st.button(" Aplică Filtrare Quantile", type="primary"):
        st.session_state['df_filtered'] = df_filtered
        st.success(f" Eliminate {n_removed} valori outlier!")

    st.markdown('<div class="sub-header">Variabile Categorice</div>', unsafe_allow_html=True)

    cat_cols = df.select_dtypes(include=['object']).columns.tolist()

    if cat_cols:

        cat_col = st.selectbox("Selectează coloana categorică:", cat_cols, key="cat_col")

        value_counts = df[cat_col].value_counts()
        value_counts_pct = (value_counts / len(df) * 100).round(2)

        fig = px.bar(
            x=value_counts.index,
            y=value_counts.values,
            labels={'x': cat_col, 'y': 'Frecvență'},
            title=f'Distribuția Categoriilor: {cat_col}',
            text=value_counts.values
        )
        fig.update_traces(textposition='outside')
        st.plotly_chart(fig, use_container_width=True)

        # Show frequency table
        freq_df = pd.DataFrame({
            'Categorie': value_counts.index,
            'Frecvență': value_counts.values,
            'Procent': value_counts_pct.values
        })
        st.dataframe(freq_df, use_container_width=True)

        # Identify rare categories
        threshold_pct = st.slider("Prag pentru categorii rare (%):", 0.1, 10.0, 1.0, 0.1)
        threshold_count = len(df) * (threshold_pct / 100)

        rare_cats = value_counts[value_counts < threshold_count]

        if len(rare_cats) > 0:
            st.warning(f" Găsite {len(rare_cats)} categorii rare (< {threshold_pct}%)")
            st.dataframe(pd.DataFrame({
                'Categorie Rară': rare_cats.index,
                'Frecvență': rare_cats.values
            }), use_container_width=True)

def show_string_processing():
    st.markdown('<h1 class="main-header"> Prelucrarea Șirurilor de Caractere</h1>', unsafe_allow_html=True)

    df = st.session_state['df'].copy()
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()

    if not cat_cols:
        st.error("Nu există coloane text în dataset!")
        return

    st.markdown('<div class="sub-header">Discretizare</div>', unsafe_allow_html=True)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if numeric_cols:
        st.markdown("### Discretizează Variabilă Continuă")

        col_to_bin = st.selectbox("Selectează coloana numerică:", numeric_cols, key="bin_col")

        col1, col2 = st.columns(2)

        with col1:
            bin_method = st.radio(
                "Metodă:",
                ['cut', 'qcut'],
                format_func=lambda x: {
                    'cut': ' cut() - Lățime Egală',
                    'qcut': ' qcut() - Frecvență Egală'
                }[x]
            )

        with col2:
            n_bins = st.slider("Număr bins:", 1, 100, 5)

        use_labels = st.checkbox("Folosește labels custom", value=False)

        if use_labels:
            labels_input = st.text_input(
                "Labels (separate prin virgulă):",
                value=f"{','.join([f'Bin_{i + 1}' for i in range(n_bins)])}"
            )
            labels = [l.strip() for l in labels_input.split(',')]

            if len(labels) != n_bins:
                st.error(f" Trebuie {n_bins} labels, ai furnizat {len(labels)}")
                labels = False
        else:
            labels = False

        if st.button(" Aplică Discretizarea", type="primary"):
            df_binned = df.copy()

            try:
                if bin_method == 'cut':
                    df_binned[f'{col_to_bin}_BINNED'] = pd.cut(
                        df_binned[col_to_bin],
                        bins=n_bins,
                        labels=labels
                    )
                else:
                    df_binned[f'{col_to_bin}_BINNED'] = pd.qcut(
                        df_binned[col_to_bin],
                        q=n_bins,
                        labels=labels,
                        duplicates='drop'
                    )

                st.session_state['df_binned'] = df_binned
                st.success("   Discretizare aplicată!")

                # Show results
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("### Histogramă Originală")
                    fig1 = px.histogram(df, x=col_to_bin, nbins=50)
                    st.plotly_chart(fig1, use_container_width=True)

                    st.metric("Valori Unice", df[col_to_bin].nunique())

                with col2:
                    st.markdown("### Distribuție Bins")
                    bin_counts = df_binned[f'{col_to_bin}_BINNED'].value_counts().sort_index()

                    fig2 = px.bar(
                        x=bin_counts.index.astype(str),
                        y=bin_counts.values,
                        labels={'x': 'Bin', 'y': 'Frecvență'},
                        text=bin_counts.values
                    )
                    fig2.update_traces(textposition='outside')
                    st.plotly_chart(fig2, use_container_width=True)

                    st.metric("Bins Create", df_binned[f'{col_to_bin}_BINNED'].nunique())

                # Show mapping
                with st.expander("      Mapping Bins"):
                    mapping_df = df_binned[[col_to_bin, f'{col_to_bin}_BINNED']].drop_duplicates().sort_values(
                        col_to_bin)
                    st.dataframe(mapping_df.head(20), use_container_width=True)


            except Exception as e:
                st.error(f" Eroare: {str(e)}")
                st.info(" Încearcă să reduci numărul de bins sau folosește qcut cu duplicates='drop'")

def show_standardization():
    st.markdown('<h1 class="main-header"> Standardizare și Normalizare</h1>', unsafe_allow_html=True)

    df = st.session_state['df'].copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if not numeric_cols:
        st.error("Nu există coloane numerice în dataset!")
        return

    st.markdown('<div class="sub-header">Metode Disponibile</div>', unsafe_allow_html=True)

    st.markdown("### Selectează Date și Metodă")

    # Select columns
    cols_to_scale = st.multiselect(
        "Selectează coloanele numerice de scalat:",
        numeric_cols,
        default=numeric_cols[:3] if len(numeric_cols) >= 3 else numeric_cols
    )

    if not cols_to_scale:
        st.warning("Selectează cel puțin o coloană!")
        return

    # Select method
    scaling_method = st.selectbox(
        "Selectează metoda de scalare:",
        ['StandardScaler', 'MinMaxScaler', 'RobustScaler', 'Normalizer', 'QuantileTransformer'],
        help="Vezi teoria pentru detalii despre fiecare metodă"
    )

    # Method-specific options
    if scaling_method == 'Normalizer':
        norm_type = st.radio(
            "Tip normă:",
            ['l1', 'l2', 'max'],
            format_func=lambda x: {
                'l1': 'L1 (Manhattan)',
                'l2': 'L2 (Euclidean)',
                'max': 'Max'
            }[x],
            horizontal=True
        )
    elif scaling_method == 'QuantileTransformer':
        n_quantiles = st.slider("Număr quantile:", 10, 1000, 100, 10)
        output_dist = st.radio(
            "Distribuție output:",
            ['uniform', 'normal'],
            horizontal=True
        )
    elif scaling_method == 'RobustScaler':
        quantile_range = st.slider(
            "Quantile range (Q1, Q3):",
            (0, 100),
            (25, 75),
            1
        )

    if st.button(" Aplică Scalarea", type="primary"):
        # Prepare data
        X = df[cols_to_scale].copy()
        X_clean = X.fillna(X.mean())  # Fill NaN pentru scalare

        # Apply scaling
        try:
            if scaling_method == 'StandardScaler':
                scaler = StandardScaler()
            elif scaling_method == 'MinMaxScaler':
                scaler = MinMaxScaler()
            elif scaling_method == 'RobustScaler':
                scaler = RobustScaler(quantile_range=quantile_range)
            elif scaling_method == 'Normalizer':
                scaler = Normalizer(norm=norm_type)
            else:  # QuantileTransformer
                scaler = QuantileTransformer(
                    n_quantiles=n_quantiles,
                    output_distribution=output_dist
                )

            X_scaled = scaler.fit_transform(X_clean)

            # Create scaled DataFrame
            df_scaled = pd.DataFrame(
                X_scaled,
                columns=[f'{col}_SCALED' for col in cols_to_scale],
                index=X.index
            )

            # Combine with original
            df_result = pd.concat([X, df_scaled], axis=1)

            st.session_state['df_scaled'] = df_result
            st.success(f"   Scalare aplicată cu {scaling_method}!")

            # Show statistics comparison
            st.markdown("### Comparație Statistici: Înainte vs După")

            stats_before = X_clean.describe()
            stats_after = df_scaled.describe()

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Înainte Scalare:**")
                st.dataframe(stats_before, use_container_width=True)

            with col2:
                st.markdown("**După Scalare:**")
                st.dataframe(stats_after, use_container_width=True)

            # Visualize distributions
            st.markdown("### Comparație Distribuții")

            for col in cols_to_scale:
                with st.expander(f"      {col}"):
                    fig = go.Figure()

                    # Original
                    fig.add_trace(go.Histogram(
                        x=X_clean[col],
                        name='Original',
                        opacity=0.7,
                        nbinsx=30
                    ))

                    # Scaled
                    fig.add_trace(go.Histogram(
                        x=df_scaled[f'{col}_SCALED'],
                        name='Scaled',
                        opacity=0.7,
                        nbinsx=30
                    ))

                    fig.update_layout(
                        title=f'Distribuție: {col}',
                        barmode='overlay',
                        xaxis_title='Valoare',
                        yaxis_title='Frecvență'
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    # Box plots
                    fig_box = go.Figure()

                    fig_box.add_trace(go.Box(
                        y=X_clean[col],
                        name='Original',
                        boxmean=True
                    ))

                    fig_box.add_trace(go.Box(
                        y=df_scaled[f'{col}_SCALED'],
                        name='Scaled',
                        boxmean=True
                    ))

                    fig_box.update_layout(title=f'Box Plot: {col}')
                    st.plotly_chart(fig_box, use_container_width=True)

        except Exception as e:
            st.error(f" Eroare la scalare: {str(e)}")
            st.info(" Verifică dacă există valori NaN sau infinite în date")

    # Comparison tool
    st.markdown('<div class="sub-header">Comparație între Metode</div>', unsafe_allow_html=True)

    with st.expander(" Compară Toate Metodele"):
        if st.button("      Generează Comparație", key="compare_methods"):
            if len(cols_to_scale) > 0:
                # Select one column for comparison
                comp_col = cols_to_scale[0]
                X_comp = df[[comp_col]].fillna(df[[comp_col]].mean())

                # Apply all methods
                methods = {
                    'Original': X_comp.values,
                    'StandardScaler': StandardScaler().fit_transform(X_comp),
                    'MinMaxScaler': MinMaxScaler().fit_transform(X_comp),
                    'RobustScaler': RobustScaler().fit_transform(X_comp),
                }

                # Create comparison plot
                fig = go.Figure()

                for method_name, data in methods.items():
                    fig.add_trace(go.Box(
                        y=data.flatten(),
                        name=method_name,
                        boxmean=True
                    ))

                fig.update_layout(
                    title=f'Comparație Metode Scalare: {comp_col}',
                    yaxis_title='Valoare',
                    showlegend=True
                )

                st.plotly_chart(fig, use_container_width=True)

                # Statistics table
                stats_comparison = pd.DataFrame({
                    method: [
                        data.min(),
                        np.percentile(data, 25),
                        np.median(data),
                        np.percentile(data, 75),
                        data.max(),
                        data.mean(),
                        data.std()
                    ]
                    for method, data in methods.items()
                }, index=['Min', 'Q1', 'Median', 'Q3', 'Max', 'Mean', 'Std'])

                st.markdown("### Statistici Comparative")
                st.dataframe(stats_comparison, use_container_width=True)

def show_descriptive_statistics():
    st.markdown('<h1 class="main-header"> Statistici Descriptive</h1>', unsafe_allow_html=True)

    df = st.session_state['df'].copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if not numeric_cols:
        st.error("Nu există coloane numerice în dataset!")
        return

    st.markdown('<div class="sub-header">Agregări</div>', unsafe_allow_html=True)

    st.markdown("### Statistici Comprehensive")

    # Full describe
    st.markdown("#### Toate Coloanele Numerice")
    desc_df = df[numeric_cols].describe()
    st.dataframe(desc_df, use_container_width=True)

    # Custom percentiles
    with st.expander("Percentile Custom"):
        percentiles_input = st.text_input(
            "Percentile (0-100, separate prin virgulă):",
            value="1, 10, 25, 50, 75, 90, 99"
        )

        try:
            percentiles = [float(p.strip()) / 100 for p in percentiles_input.split(',')]
            custom_desc = df[numeric_cols].describe(percentiles=percentiles)
            st.dataframe(custom_desc, use_container_width=True)
        except:
            st.error("Format invalid! Folosește numere separate prin virgulă.")

    st.markdown('<div class="sub-header">Skewness</div>', unsafe_allow_html=True)

    st.markdown("### Calculează Skewness")

    # Calculate skewness
    skewness = df[numeric_cols].skew()
    skewness_df = pd.DataFrame({
        'Coloană': skewness.index,
        'Skewness': skewness.values,
        'Interpretare': skewness.apply(lambda x:
                                       'Simetrică' if abs(x) < 0.5 else
                                       'Moderat înclinată dreapta' if 0.5 <= x < 1 else
                                       'Foarte înclinată dreapta' if x >= 1 else
                                       'Moderat înclinată stânga' if -1 < x <= -0.5 else
                                       'Foarte înclinată stânga'
                                       ).values
    }).sort_values('Skewness', key=abs, ascending=False)

    col1, col2 = st.columns([2, 1])

    with col1:
        # Plot skewness
        fig = px.bar(
            skewness_df,
            x='Coloană',
            y='Skewness',
            color='Interpretare',
            title='Skewness pe Coloane',
            text='Skewness'
        )
        fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
        fig.add_hline(y=0, line_dash="dash", line_color="black")
        fig.add_hline(y=0.5, line_dash="dot", line_color="green", annotation_text="Moderate threshold")
        fig.add_hline(y=-0.5, line_dash="dot", line_color="green")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### Tabel Skewness")
        st.dataframe(skewness_df, use_container_width=True)

    # Visualize most skewed
    most_skewed = skewness_df.iloc[0]['Coloană']

    with st.expander(f"      Vizualizare: {most_skewed} (Cea Mai Înclinată)"):
        fig = go.Figure()

        fig.add_trace(go.Histogram(
            x=df[most_skewed].dropna(),
            nbinsx=50,
            name='Histogramă'
        ))

        # Add mean and median lines
        mean_val = df[most_skewed].mean()
        median_val = df[most_skewed].median()

        fig.add_vline(x=mean_val, line_dash="dash", line_color="red", annotation_text=f"Medie: {mean_val:.2f}")
        fig.add_vline(x=median_val, line_dash="dash", line_color="blue",
                      annotation_text=f"Mediană: {median_val:.2f}")

        fig.update_layout(title=f'Distribuție {most_skewed} (Skewness: {skewness_df.iloc[0]["Skewness"]:.3f})')
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="sub-header">Kurtosis</div>', unsafe_allow_html=True)

    st.markdown("### Calculează Kurtosis")

    # Calculate kurtosis
    kurt = df[numeric_cols].kurtosis()  # Excess kurtosis (Fisher)
    kurt_df = pd.DataFrame({
        'Coloană': kurt.index,
        'Excess Kurtosis': kurt.values,
        'Interpretare': kurt.apply(lambda x:
                                   'Mesokurtic (Normal)' if -0.5 <= x <= 0.5 else
                                   'Leptokurtic (Cozi grele)' if x > 0.5 else
                                   'Platykurtic (Cozi ușoare)'
                                   ).values
    }).sort_values('Excess Kurtosis', key=abs, ascending=False)

    col1, col2 = st.columns([2, 1])

    with col1:
        fig = px.bar(
            kurt_df,
            x='Coloană',
            y='Excess Kurtosis',
            color='Interpretare',
            title='Kurtosis pe Coloane',
            text='Excess Kurtosis'
        )
        fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
        fig.add_hline(y=0, line_dash="dash", line_color="black")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### Tabel Kurtosis")
        st.dataframe(kurt_df, use_container_width=True)

    st.markdown('<div class="sub-header">Matricea de corelație</div>', unsafe_allow_html=True)

    st.markdown("### Matrice de Corelație")

    # Select correlation method
    corr_method = st.radio(
        "Metoda de corelație:",
        ['pearson', 'spearman', 'kendall'],
        format_func=lambda x: {
            'pearson': 'Pearson (Linear)',
            'spearman': 'Spearman (Rank)',
            'kendall': 'Kendall (Rank)'
        }[x],
        horizontal=True
    )

    # Calculate correlation
    corr_matrix = df[numeric_cols].corr(method=corr_method)

    # Heatmap
    fig = px.imshow(
        corr_matrix,
        text_auto='.2f',
        aspect='auto',
        color_continuous_scale='RdBu_r',
        color_continuous_midpoint=0,
        title=f'Heatmap Corelație ({corr_method.capitalize()})'
    )
    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig, use_container_width=True)

    # Strong correlations
    st.markdown("### Corelații Puternice (|r| > 0.7)")

    # Get upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    corr_upper = corr_matrix.where(mask)

    # Flatten and filter
    strong_corr = []
    for col in corr_upper.columns:
        for idx in corr_upper.index:
            val = corr_upper.loc[idx, col]
            if not pd.isna(val) and abs(val) > 0.7:
                strong_corr.append({
                    'Variabila 1': idx,
                    'Variabila 2': col,
                    'Corelație': val,
                    'Forță': 'Foarte Puternică' if abs(val) > 0.9 else 'Puternică'
                })

    if strong_corr:
        strong_corr_df = pd.DataFrame(strong_corr).sort_values('Corelație', key=abs, ascending=False)
        st.dataframe(strong_corr_df, use_container_width=True)


def show_graphical_representations():
    st.markdown('<h1 class="main-header"> Reprezentari grafice</h1>', unsafe_allow_html=True)

    df = st.session_state['df'].copy()

    # Găsește coloane necesare
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()

    st.markdown('<div class="sub-header">Box Plot - Comparatii pe grupuri</div>', unsafe_allow_html=True)

    if not numeric_cols:
        st.error("Nu există coloane numerice în dataset!")
        return

    if not cat_cols:
        st.error("Nu există coloane categorice în dataset!")
        return

    col1, col2 = st.columns(2)

    with col1:
        box_y = st.selectbox("Variabila numerică (Y):", numeric_cols, key="box_y")

    with col2:
        box_x = st.selectbox("Grupare (X):", cat_cols, key="box_x")

    box_color = st.selectbox(
        "Culoare pe categorie (opțional):",
        [None] + cat_cols,
        key="box_color"
    )

    try:
        fig = px.box(
            df,
            x=box_x,
            y=box_y,
            color=box_color if box_color else box_x,
            title=f'Box Plot: {box_y} pe {box_x}',
            points='outliers'
        )
        st.plotly_chart(fig, use_container_width=True)

        # Statistics by group
        with st.expander("Statistici pe Grupuri"):
            stats = df.groupby(box_x)[box_y].describe()
            st.dataframe(stats, use_container_width=True)

    except Exception as e:
        st.error(f"Eroare la generarea graficului: {str(e)}")
        st.info("Verifica daca exista valori valide in coloanele selectate.")


def show_problem_setup():
    st.markdown('<h1 class="main-header"> Problem Setup</h1>', unsafe_allow_html=True)

    if 'df' not in st.session_state:
        st.warning("⚠️ Încarcă mai întâi un dataset.")
        return

    df = st.session_state['df']

    if df.empty:
        st.error("Dataset-ul este gol.")
        return

    st.markdown('<div class="sub-header">Definirea Problemei ML</div>', unsafe_allow_html=True)

    st.markdown("### 1️⃣ Selectare Target")

    target_col = st.selectbox(
        "Alege coloana țintă (target):",
        options=list(df.columns),
        index=None,
        placeholder="Selectează coloana target"
    )

    if target_col is None:
        st.info("⬆️ Selectează coloana target pentru a continua.")
        return

    unique_vals = df[target_col].dropna().nunique()

    if unique_vals <= 10:
        problem_type = "Clasificare"
    else:
        problem_type = "Regresie"

    st.success(f"🧠 Tip problemă detectat automat: **{problem_type}**")

    st.markdown("### 2️⃣ Selectare Feature-uri")

    all_features = [col for col in df.columns if col != target_col]

    select_all = st.checkbox(
        "Selectează toate feature-urile",
        value=True
    )

    if select_all:
        selected_features = all_features
    else:
        selected_features = st.multiselect(
            "Alege feature-urile:",
            options=all_features,
            default=all_features
        )

    if not selected_features:
        st.warning("⚠️ Selectează cel puțin un feature.")
        return

    st.markdown("### 3️⃣ Excludere Coloane (opțional)")

    excluded_cols = st.multiselect(
        "Coloane de exclus din feature-uri:",
        options=selected_features
    )

    final_features = [col for col in selected_features if col not in excluded_cols]

    if not final_features:
        st.error("❌ Toate feature-urile au fost excluse.")
        return

    st.markdown('<div class="sub-header"> Rezumat Configurație</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Target", target_col)

    with col2:
        st.metric("Tip Problemă", problem_type)

    with col3:
        st.metric("Nr. Feature-uri", len(final_features))

    st.markdown("### Feature-uri Finale")
    st.dataframe(
        pd.DataFrame({"Feature": final_features}),
        use_container_width=True
    )

    st.session_state['target'] = target_col
    st.session_state['features'] = final_features
    st.session_state['problem_type'] = problem_type

    st.success("✅ Problem Setup salvat cu succes!")

def show_preprocessing_pipeline():
    st.markdown('<h1 class="main-header"> Preprocesare & Pipeline</h1>', unsafe_allow_html=True)

    required_keys = ['df', 'target', 'features', 'problem_type']
    if not all(k in st.session_state for k in required_keys):
        st.warning("Finalizează mai întâi Problem Setup.")
        return

    df = st.session_state['df']
    target = st.session_state['target']
    features = st.session_state['features']
    problem_type = st.session_state['problem_type']

    X = df[features]
    y = df[target]

    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()

    st.markdown("### Opțiuni Preprocesare")

    col1, col2 = st.columns(2)

    with col1:
        num_imputer = st.selectbox(
            "Imputare numerică:",
            ["mean", "median", "most_frequent"]
        )

        scaler_option = st.selectbox(
            "Scalare numerică:",
            ["StandardScaler", "MinMaxScaler", "Fără scalare"]
        )

    with col2:
        cat_imputer = st.selectbox(
            "Imputare categorică:",
            ["most_frequent"]
        )

        use_outliers = st.checkbox("Eliminare outlieri (IQR simplu)")
        use_feature_selection = st.checkbox("Selecție feature-uri (SelectKBest)")

    if use_outliers and numeric_features:
        Q1 = X[numeric_features].quantile(0.25)
        Q3 = X[numeric_features].quantile(0.75)
        IQR = Q3 - Q1

        mask = ~((X[numeric_features] < (Q1 - 1.5 * IQR)) |
                 (X[numeric_features] > (Q3 + 1.5 * IQR))).any(axis=1)

        X = X.loc[mask]
        y = y.loc[mask]

        st.info(f"🧹 Outlieri eliminați. Rânduri rămase: {len(X)}")

    num_steps = [("imputer", SimpleImputer(strategy=num_imputer))]

    if scaler_option == "StandardScaler":
        num_steps.append(("scaler", StandardScaler()))
    elif scaler_option == "MinMaxScaler":
        num_steps.append(("scaler", MinMaxScaler()))

    numeric_pipeline = Pipeline(num_steps)

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features)
    ])
    steps = [("preprocessor", preprocessor)]

    if use_feature_selection:
        k = st.slider("Număr feature-uri selectate (K):", 1, 50, 10)

        selector = SelectKBest(
            score_func=f_classif if problem_type == "Clasificare" else f_regression,
            k=k
        )
        steps.append(("feature_selection", selector))

    full_pipeline = Pipeline(steps)

    test_size = st.slider("Test size:", 0.1, 0.5, 0.2)

    if problem_type == "Clasificare" and y.value_counts().min() >= 2:
        stratify_option = y
    else:
        stratify_option = None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
        stratify=stratify_option
    )

    if st.button(" Aplică Pipeline", type="primary"):
        full_pipeline.fit(X_train, y_train)

        st.session_state['pipeline'] = full_pipeline
        st.session_state['X_train'] = X_train
        st.session_state['X_test'] = X_test
        st.session_state['y_train'] = y_train
        st.session_state['y_test'] = y_test

        st.success(" Pipeline antrenat și salvat!")

        X_train_transformed = full_pipeline.transform(X_train)

        st.markdown("### Rezultat Preprocesare")
        st.write("Dimensiune X_train după pipeline:", X_train_transformed.shape)

def show_train_test_split():
    st.markdown('<h1 class="main-header"> Train / Test Split</h1>', unsafe_allow_html=True)

    required_keys = ['df', 'target', 'features', 'problem_type']
    if not all(k in st.session_state for k in required_keys):
        st.warning("Finalizează mai întâi Problem Setup.")
        return

    df = st.session_state['df']
    target = st.session_state['target']
    features = st.session_state['features']
    problem_type = st.session_state['problem_type']

    X = df[features]
    y = df[target]

    st.markdown("### Opțiuni Split")

    split_type = st.radio(
        "Tip split:",
        ["Train / Test", "Train / Validation / Test"],
        key="split_type_radio"
    )

    random_state = st.number_input(
        "Random state:",
        min_value=0,
        max_value=9999,
        value=42,
        step=1,
        key="random_state_input"
    )

    if problem_type == "Clasificare" and y.value_counts().min() >= 2:
        stratify_option = y
        st.info("Stratificare activă (clasificare).")
    else:
        stratify_option = None
        st.info("Fără stratificare (regresie sau clase rare).")

    if split_type == "Train / Test":
        test_size = st.slider(
            "Proporție Test:",
            0.1,
            0.5,
            0.2,
            key="test_size_split"
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_option
        )

        st.session_state['X_train'] = X_train
        st.session_state['X_test'] = X_test
        st.session_state['y_train'] = y_train
        st.session_state['y_test'] = y_test

        st.success("Split Train / Test realizat!")


    else:

        test_size = st.slider(

            "Proporție Test:",

            0.1,

            0.4,

            0.2,

            key="test_size_split_3way"

        )

        val_size = st.slider(

            "Proporție Validation:",

            0.1,

            0.4,

            0.2,

            key="val_size_split"

        )


        X_temp, X_test, y_temp, y_test = train_test_split(

            X,

            y,

            test_size=test_size,

            random_state=random_state,

            stratify=y if stratify_option is not None else None

        )

        val_ratio_adjusted = val_size / (1 - test_size)


        X_train, X_val, y_train, y_val = train_test_split(

            X_temp,

            y_temp,

            test_size=val_ratio_adjusted,

            random_state=random_state,

            stratify=y_temp if stratify_option is not None else None

        )

        st.session_state['X_train'] = X_train

        st.session_state['X_val'] = X_val

        st.session_state['X_test'] = X_test

        st.session_state['y_train'] = y_train

        st.session_state['y_val'] = y_val

        st.session_state['y_test'] = y_test

        st.success("Split Train / Validation / Test realizat!")

    st.markdown("### Dimensiuni seturi")
    st.write("Train:", len(st.session_state['X_train']))
    if 'X_val' in st.session_state:
        st.write("Validation:", len(st.session_state['X_val']))
    st.write("Test:", len(st.session_state['X_test']))

def show_models():
    st.markdown('<h1 class="main-header"> Models</h1>', unsafe_allow_html=True)

    required_keys = ['pipeline', 'X_train', 'y_train', 'problem_type']
    if not all(k in st.session_state for k in required_keys):
        st.warning("Finalizează mai întâi Split + Preprocesare.")
        return

    base_pipeline = st.session_state['pipeline']
    X_train = st.session_state['X_train']
    y_train = st.session_state['y_train']
    problem_type = st.session_state['problem_type']

    st.markdown("### Selectare Algoritmi")

    if problem_type == "Clasificare":
        available_models = [
            "Logistic Regression",
            "Random Forest Classifier",
            "SVM (SVC)"
        ]
    else:
        available_models = [
            "Linear Regression",
            "Ridge Regression",
            "Random Forest Regressor"
        ]

    selected_models = st.multiselect(
        "Alege algoritmi:",
        options=available_models,
        key="models_multiselect"
    )

    if not selected_models:
        st.info("Selectează cel puțin un algoritm.")
        return

    st.markdown("### Hiperparametri")
    model_configs = {}

    for model_name in selected_models:
        st.subheader(model_name)

        if model_name == "Logistic Regression":
            C = st.slider(
                "C (regularizare)",
                0.01, 10.0, 1.0,
                key=f"logreg_c_{model_name}"
            )
            max_iter = st.slider(
                "max_iter",
                100, 1000, 300,
                key=f"logreg_iter_{model_name}"
            )

            model_configs[model_name] = LogisticRegression(
                C=C,
                max_iter=max_iter
            )

        elif model_name == "Random Forest Classifier":
            n_estimators = st.slider(
                "n_estimators",
                50, 300, 100,
                key=f"rfc_estimators_{model_name}"
            )
            max_depth = st.slider(
                "max_depth",
                2, 20, 10,
                key=f"rfc_depth_{model_name}"
            )

            model_configs[model_name] = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42
            )

        elif model_name == "SVM (SVC)":
            C = st.slider(
                "C",
                0.1, 10.0, 1.0,
                key=f"svc_c_{model_name}"
            )
            kernel = st.selectbox(
                "kernel",
                ["linear", "rbf"],
                key=f"svc_kernel_{model_name}"
            )

            model_configs[model_name] = SVC(
                C=C,
                kernel=kernel,
                probability=True
            )

        elif model_name == "Linear Regression":
            model_configs[model_name] = LinearRegression()

        elif model_name == "Ridge Regression":
            alpha = st.slider(
                "alpha",
                0.01, 10.0, 1.0,
                key=f"ridge_alpha_{model_name}"
            )

            model_configs[model_name] = Ridge(alpha=alpha)

        elif model_name == "Random Forest Regressor":
            n_estimators = st.slider(
                "n_estimators",
                50, 300, 100,
                key=f"rfr_estimators_{model_name}"
            )
            max_depth = st.slider(
                "max_depth",
                2, 20, 10,
                key=f"rfr_depth_{model_name}"
            )

            model_configs[model_name] = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42
            )

    # -------------------- TRAIN MODELS --------------------
    if st.button("🚀 Train Models", type="primary", key="train_models_btn"):

        trained_models = []
        results = []

        X_test = st.session_state["X_test"]
        y_test = st.session_state["y_test"]

        for name, model in model_configs.items():

            full_pipeline = Pipeline([
                ("preprocessor", base_pipeline),
                ("model", model)
            ])

            full_pipeline.fit(X_train, y_train)
            y_pred = full_pipeline.predict(X_test)

            trained_models.append({
                "Model": name,
                "Pipeline": full_pipeline
            })

            # -------- METRICI --------
            if problem_type == "Clasificare":
                row = {
                    "Model": name,
                    "Accuracy": accuracy_score(y_test, y_pred),
                    "Precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
                    "Recall": recall_score(y_test, y_pred, average="weighted"),
                    "F1": f1_score(y_test, y_pred, average="weighted")
                }

                if y_test.nunique() == 2 and hasattr(full_pipeline, "predict_proba"):
                    y_proba = full_pipeline.predict_proba(X_test)[:, 1]
                    row["ROC-AUC"] = roc_auc_score(y_test, y_proba)
                else:
                    row["ROC-AUC"] = None

            else:
                row = {
                    "Model": name,
                    "MAE": mean_absolute_error(y_test, y_pred),
                    "RMSE": mean_squared_error(y_test, y_pred, squared=False),
                    "R2": r2_score(y_test, y_pred)
                }

            results.append(row)

        # -------------------- DATAFRAME CURAT --------------------
        results_df = pd.DataFrame(results)

        if problem_type == "Clasificare":
            cols = ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
        else:
            cols = ["Model", "MAE", "RMSE", "R2"]

        results_df = results_df[cols]
        results_df[cols[1:]] = results_df[cols[1:]].astype(float).round(3)

        st.session_state["trained_models"] = {
            m["Model"]: m["Pipeline"] for m in trained_models
        }

        st.subheader("📊 Rezultate modele")
        st.dataframe(results_df, use_container_width=True)

        st.success("✅ Modelele au fost antrenate și evaluate!")


def show_evaluation():

    st.header("📊 Evaluation & Model Comparison")

    if "trained_models" not in st.session_state:
        st.warning("⚠️ Antrenează modelele mai întâi.")
        return

    X_test = st.session_state["X_test"]
    y_test = st.session_state["y_test"]
    trained_models = st.session_state["trained_models"]
    problem_type = st.session_state["problem_type"]

    results = []

    for name, pipeline in trained_models.items():

        y_pred = pipeline.predict(X_test)

        if problem_type == "Clasificare":

            row = {
                "Model": name,
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
                "Recall": recall_score(y_test, y_pred, average="weighted"),
                "F1": f1_score(y_test, y_pred, average="weighted")
            }

            if y_test.nunique() == 2 and hasattr(pipeline, "predict_proba"):
                y_proba = pipeline.predict_proba(X_test)[:, 1]
                row["ROC-AUC"] = roc_auc_score(y_test, y_proba)

        else:
            row = {
                "Model": name,
                "MAE": mean_absolute_error(y_test, y_pred),
                "RMSE": mean_squared_error(y_test, y_pred, squared=False),
                "R2": r2_score(y_test, y_pred)
            }

        results.append(row)

    results_df = pd.DataFrame(results)

    st.subheader("📋 Metrici pe setul de test")

    metric_cols = results_df.columns.drop("Model")
    st.dataframe(
        results_df.style.format({c: "{:.3f}" for c in metric_cols}),
        use_container_width=True
    )

    if problem_type == "Clasificare":
        metric = st.selectbox(
            "Metrică pentru best model",
            [c for c in results_df.columns if c != "Model"]
        )
        best_idx = results_df[metric].idxmax()
    else:
        metric = st.selectbox("Metrică pentru best model", ["R2", "RMSE", "MAE"])
        best_idx = results_df[metric].idxmax() if metric == "R2" else results_df[metric].idxmin()

    best_model = results_df.loc[best_idx, "Model"]
    best_value = results_df.loc[best_idx, metric]

    st.success(
        f"🏆 **Best model:** {best_model}\n\n"
        f"📊 **{metric} = {best_value:.3f}**"
    )

    if problem_type == "Clasificare":
        if metric == "Recall":
            st.info(
                "🔍 **Recall** măsoară cât de bine sunt identificate cazurile pozitive.\n"
                "În acest context: cât de bine detectează modelul clienții care intră în default."
            )
        elif metric == "Precision":
            st.info(
                "🎯 **Precision** arată cât de corecte sunt predicțiile pozitive.\n"
                "Când modelul spune *default*, cât de des are dreptate."
            )
        elif metric == "Accuracy":
            st.info(
                "📈 **Accuracy** reprezintă proporția totală de predicții corecte."
            )
        elif metric == "F1":
            st.info(
                "⚖️ **F1-score** este un echilibru între Precision și Recall."
            )

    if problem_type == "Clasificare":
        st.subheader("🔍 Confusion Matrix")

        model_name = st.selectbox(
            "Alege model pentru Confusion Matrix",
            list(trained_models.keys()),
            key="cm_model_select"
        )

        cm = confusion_matrix(
            y_test,
            trained_models[model_name].predict(X_test)
        )

        fig, ax = plt.subplots()
        ax.imshow(cm)
        ax.set_title(f"Confusion Matrix — {model_name}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, cm[i, j], ha="center", va="center")

        st.pyplot(fig)

if __name__ == "__main__":
    selected_module = sidebar_navigation()

    if selected_module == " Incarcare date":
        show_data_connection()
    elif selected_module == " Filtrare date":
        show_data_filtering()
    elif selected_module == " Analiza valori lipsa":
        show_missing_values_analysis()
    elif selected_module == " Detectarea valorilor anormale":
        show_outlier_detection()
    elif selected_module == " Prelucrarea sirurilor de caractere":
        show_string_processing()
    elif selected_module == " Standardizare si normalizare":
        show_standardization()
    elif selected_module == " Statistici descriptive":
        show_descriptive_statistics()
    elif selected_module == " Reprezentari grafice":
        show_graphical_representations()
    elif selected_module == " Problem Setup - ML":
        show_problem_setup()
    elif selected_module == " Preprocesare & Pipeline - ML":
        show_preprocessing_pipeline()
    elif selected_module == " Train/Test Split - ML":
        show_train_test_split()
    elif selected_module == " Models - ML":
        show_models()
    elif selected_module == " Evaluare si comparare - ML":
        show_evaluation()



