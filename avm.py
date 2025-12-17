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

st.markdown("""
<style>
    body, .stApp {
        background-color: #4179AF;  
    }
    .main-header, .sub-header, .method-box, .why-box, .theory-box, .code-explanation, .success-box, 
    .stMarkdown, .stText {
        color: #DA70D6;  
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def load_data_from_csv(uploaded_file):
    """Încarcă datele dintr-un fișier CSV încărcat în Streamlit."""
    df = pd.read_csv(uploaded_file)
    return df
def sidebar_navigation():
    st.sidebar.markdown("#  Proiect MAIA")
    st.sidebar.markdown("### Navighează:")

    sections = [
        " Incarcare date",
        " Curatarea datelor",
        " Detectarea valorilor anormale",
        " Prelucrarea sirurilor de caractere",
        " Standardizare si normalizare",
        " Statistici descriptive",
        " Reprezentari grafice"
    ]

    selected = st.sidebar.radio("Selectează Modulul:", sections)

    st.sidebar.markdown("---")

    return selected

# Date
def show_data_connection():
    st.markdown('<h1 class="main-header"> Încărcare Date</h1>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Încarcă datele dintr-un fișier CSV</div>', unsafe_allow_html=True)

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

# Curățarea Datelor
def show_data_cleaning():
    st.markdown('<h1 class="main-header"> Curățarea Datelor</h1>', unsafe_allow_html=True)

    df = st.session_state['df'].copy()

    # Eliminarea duplicatelor
    st.markdown('<div class="sub-header">Eliminarea Duplicatelor</div>', unsafe_allow_html=True)
    st.markdown("### Găsește și Elimină Duplicate")

    # Select columns for duplicate check
    all_cols = df.columns.tolist()
    default_cols = ['ID_CLIENT'] if 'ID_CLIENT' in all_cols else [all_cols[0]]

    duplicate_cols = st.multiselect(
        "Selectează coloanele pentru verificarea duplicatelor:",
        all_cols,
        default=default_cols
    )

    if duplicate_cols:
        # Find duplicates
        duplicates = df[df.duplicated(subset=duplicate_cols, keep=False)]
        n_duplicates = len(df[df.duplicated(subset=duplicate_cols)])

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(" Total Rânduri", len(df))

        with col2:
            st.metric(" Duplicate Găsite", n_duplicates)

        with col3:
            pct_dup = (n_duplicates / len(df) * 100) if len(df) > 0 else 0
            st.metric(" Procent Duplicate", f"{pct_dup:.2f}%")

        if n_duplicates > 0:
            st.warning(f"⚠ Găsite {n_duplicates} rânduri duplicate!")

            with st.expander(" Vezi Duplicate"):
                st.dataframe(duplicates.sort_values(by=duplicate_cols).head(20), use_container_width=True)

            # Options for handling duplicates
            keep_option = st.radio(
                "Ce apariție vrei să păstrezi?",
                ['first', 'last', False],
                format_func=lambda x: {
                    'first': 'Prima apariție',
                    'last': 'Ultima apariție',
                    False: 'Elimină toate (nu păstra nimic)'
                }[x],
                horizontal=True
            )

            if st.button(" Elimină Duplicate", type="primary"):
                df_clean = df.drop_duplicates(subset=duplicate_cols, keep=keep_option)
                n_removed = len(df) - len(df_clean)

                st.success(f" Eliminate {n_removed} rânduri duplicate!")
                st.metric(" Rânduri Rămase", len(df_clean))

                # Store cleaned data
                st.session_state['df_clean'] = df_clean

                # Show before/after
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**Înainte:**")
                    st.dataframe(df.head(10), use_container_width=True)

                with col2:
                    st.markdown("**După:**")
                    st.dataframe(df_clean.head(10), use_container_width=True)

# Tratare completă (toate coloanele)
    st.markdown("### Tratare Automată pentru Toate Coloanele")

    with st.expander("Aplică Strategie Globală"):
        st.markdown("""
        Aplică o strategie de tratare pentru toate coloanele cu valori lipsă simultan.
        """)

        global_strategy = st.radio(
            "Strategie globală:",
            ['smart', 'drop_rows', 'drop_cols'],
            format_func=lambda x: {
                'smart': ' Smart (Medie pt numeric, Mod pt categoric)',
                'drop_rows': ' Elimină Rânduri cu NaN',
                'drop_cols': 'Elimină Coloane cu > 30% NaN'
            }[x]
        )

        if st.button(" Aplică Tratare Globală", type="primary"):
            df_global = df.copy()

            if global_strategy == 'smart':
                # Numeric columns - mean
                numeric_cols = df_global.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    if df_global[col].isnull().sum() > 0:
                        df_global[col].fillna(df_global[col].mean(), inplace=True)

                # Categorical columns - mode
                cat_cols = df_global.select_dtypes(include=['object']).columns
                for col in cat_cols:
                    if df_global[col].isnull().sum() > 0:
                        mode_val = df_global[col].mode()[0] if len(df_global[col].mode()) > 0 else 'MISSING'
                        df_global[col].fillna(mode_val, inplace=True)

                st.success("Aplicată")

            elif global_strategy == 'drop_rows':
                df_global.dropna(inplace=True)
                n_dropped = len(df) - len(df_global)
                st.success(f" Eliminate {n_dropped} rânduri!")

            else:  # drop_cols
                threshold = 0.3
                for col in df_global.columns:
                    if (df_global[col].isnull().sum() / len(df_global)) > threshold:
                        df_global.drop(columns=[col], inplace=True)
                n_dropped_cols = len(df.columns) - len(df_global.columns)
                st.success(f" Eliminate {n_dropped_cols} coloane!")

            # Store and show results
            st.session_state['df_global_clean'] = df_global

            col1, col2 = st.columns(2)

            with col1:
                st.metric("Înainte - Total NaN", df.isnull().sum().sum())
                st.metric("Înainte - Dimensiune", f"{df.shape[0]} × {df.shape[1]}")

            with col2:
                st.metric("După - Total NaN", df_global.isnull().sum().sum())
                st.metric("După - Dimensiune", f"{df_global.shape[0]} × {df_global.shape[1]}")

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
    st.markdown('<div class="sub-header">Label Encoding</div>', unsafe_allow_html=True)

    st.markdown("### Transformă Categorii în Numere")

    cols_to_encode = st.multiselect(
        "Selectează coloanele de transformat:",
        cat_cols,
        default=cat_cols[:2] if len(cat_cols) >= 2 else cat_cols
    )

    if cols_to_encode and st.button(" Aplică Label Encoding", type="primary"):
        df_encoded = df.copy()

        mappings = {}

        for col in cols_to_encode:
            le = LabelEncoder()
            df_encoded[f'{col}_ENCODED'] = le.fit_transform(df_encoded[col].astype(str))

            # Store mapping
            mappings[col] = dict(zip(le.classes_, le.transform(le.classes_)))

        st.session_state['df_encoded'] = df_encoded
        st.success(f"   Transformate {len(cols_to_encode)} coloane!")

        # Show results
        for col in cols_to_encode:
            with st.expander(f"      Mapare: {col}"):
                st.markdown(f"### {col} → {col}_ENCODED")

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**Original:**")
                    st.dataframe(df[col].value_counts(), use_container_width=True)

                with col2:
                    st.markdown("**Encoded:**")
                    mapping_df = pd.DataFrame({
                        'Categorie': list(mappings[col].keys()),
                        'Cod': list(mappings[col].values())
                    }).sort_values('Cod')
                    st.dataframe(mapping_df, use_container_width=True)

                # Sample comparison
                st.markdown("**Exemplu Transformare:**")
                sample_df = pd.DataFrame({
                    'Original': df[col].head(10),
                    'Encoded': df_encoded[f'{col}_ENCODED'].head(10)
                })
                st.dataframe(sample_df, use_container_width=True)


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

if __name__ == "__main__":
    selected_module = sidebar_navigation()

    if selected_module == " Incarcare date":
        show_data_connection()
    elif selected_module == " Curatarea datelor":
        show_data_cleaning()
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
