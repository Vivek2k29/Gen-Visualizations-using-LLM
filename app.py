import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import re
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import os

warnings.filterwarnings('ignore')

# Try optional PDF library
try:
    import pdfplumber
    pdfplumber_available = True
except Exception:
    pdfplumber_available = False

# -------------------------
# Page config
# -------------------------
st.set_page_config(
    page_title="GenVis - Ultimate Chart Generator",
    page_icon="📊",
    layout="wide"
)

st.title("📊 GenVis - Ultimate Accurate Chart Generator")
st.markdown("**Precision visualization** with **manual controls** for perfect accuracy!")

# -------------------------
# Session State
# -------------------------
if 'df' not in st.session_state:
    st.session_state.df = None
if 'chart_generated' not in st.session_state:
    st.session_state.chart_generated = False
if 'manual_mode' not in st.session_state:
    st.session_state.manual_mode = False

# -------------------------
# Utility: cached file loader (supports CSV, Excel, PDF)
# -------------------------
@st.cache_data(show_spinner=False)
def load_file_bytes(uploaded_file_bytes, filename: str):
    """Return DataFrame from uploaded bytes and filename extension."""
    ext = os.path.splitext(filename)[1].lower()
    if ext == '.csv':
        return pd.read_csv(io.BytesIO(uploaded_file_bytes))
    elif ext in ['.xls', '.xlsx']:
        return pd.read_excel(io.BytesIO(uploaded_file_bytes))
    elif ext == '.pdf':
        if not pdfplumber_available:
            raise Exception("pdfplumber not installed on server. Add pdfplumber to requirements to enable PDF parsing.")
        # attempt to get the first table from any page
        with pdfplumber.open(io.BytesIO(uploaded_file_bytes)) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                if tables and len(tables) > 0:
                    table = tables[0]
                    df = pd.DataFrame(table[1:], columns=table[0])
                    # try converting numeric-like columns
                    for c in df.columns:
                        try:
                            df[c] = pd.to_numeric(df[c].astype(str).str.replace(',', ''), errors='ignore')
                        except Exception:
                            pass
                    return df
            # fallback: try to read all text and parse as CSV-like (best-effort)
            text = "\n".join([p.extract_text() or "" for p in pdf.pages])
            try:
                return pd.read_csv(io.StringIO(text))
            except Exception:
                raise Exception("No table found in PDF (or couldn't parse). Provide CSV/Excel for best results.")
    else:
        raise Exception("Unsupported file type. Supported: .csv, .xls, .xlsx, .pdf")


# -------------------------
# ULTIMATE CHART DETECTOR with caching & enriched docs
# -------------------------
@st.cache_resource
def get_detector_resources():
    """
    Build combined pattern docs + TF-IDF vectorizer once (cached) for faster semantic matching.
    Returns the chart_patterns dict, pattern_docs list, vectorizer and tfidf_matrix.
    """
    # Chart patterns (kept concise here; main app still uses UltimateChartDetector for keywords)
    base_patterns = {
        'pie': [
            'pie chart percentage distribution share breakdown category proportion what percent'
        ],
        'bar': [
            'bar chart comparison ranking compare categories across groups by category values'
        ],
        'scatter': [
            'scatter plot correlation relationship x vs y numeric bivariate points'
        ],
        'line': [
            'line chart trend over time time series plot date monthly yearly progression'
        ],
        'histogram': [
            'histogram distribution frequency bins value distribution spread'
        ],
        'box': [
            'box plot boxplot median quartile outlier distribution statistical summary'
        ],
        'heatmap': [
            'heatmap correlation matrix color-coded relationship multiple variables'
        ],
        'area': [
            'area chart stacked cumulative over time total over time filled curve'
        ]
    }

    # Add multiple short example phrases per chart-type to enrich semantics
    example_phrases = {
        'pie': [
            'what percentage of sales comes from each category',
            'show distribution of product categories as percentages',
            'category share of total'
        ],
        'bar': [
            'compare revenue across regions by bar chart',
            'ranking of top products by sales',
            'bar chart of average score per group'
        ],
        'scatter': [
            'plot price versus demand to see correlation',
            'relationship between height and weight points',
            'scatter of temperature against humidity'
        ],
        'line': [
            'time series of revenue over the last year',
            'monthly traffic trend line plot',
            'show growth over time'
        ],
        'histogram': [
            'age distribution histogram',
            'frequency of scores across bins',
            'histogram of test results'
        ],
        'box': [
            'show quartiles and outliers for exam scores',
            'boxplot of salary by department',
            'statistical summary plot'
        ],
        'heatmap': [
            'correlation heatmap of numeric features',
            'show correlation matrix',
            'heatmap of variable relationships'
        ],
        'area': [
            'cumulative sales over time area chart',
            'stacked area of categories over months',
            'filled line showing totals over time'
        ]
    }

    pattern_docs = []
    pattern_keys = []
    for k, base in base_patterns.items():
        # combine base + examples into single doc string
        combined = " ".join(base + example_phrases.get(k, []))
        pattern_docs.append(combined)
        pattern_keys.append(k)

    # Build TF-IDF vectorizer
    try:
        vectorizer = TfidfVectorizer(ngram_range=(1,2), stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(pattern_docs)
    except Exception:
        vectorizer = None
        tfidf_matrix = None

    return {
        'pattern_keys': pattern_keys,
        'pattern_docs': pattern_docs,
        'vectorizer': vectorizer,
        'tfidf_matrix': tfidf_matrix
    }


class UltimateChartDetector:
    def __init__(self):
        # Keep your extensive keyword lists but rely on cached resources for TF-IDF
        self.chart_patterns = {
            'pie': {
                'primary': [
                    'pie', 'pie chart', 'percentage', 'percent', 'proportion', 'share', 'breakdown',
                    'composition', 'distribution of', 'ratio', 'fraction', 'slices', 'wedges', 'segments',
                    'market share', 'allocation', 'category share', 'portion'
                ],
                'secondary': ['whole', 'entirety', 'total', '100%', 'divided into', 'grouped by'],
                'context': ['category', 'group', 'type']
            },
            'bar': {
                'primary': ['bar', 'bar chart', 'compare', 'comparison', 'ranking', 'by category', 'across'],
                'secondary': ['value', 'amount', 'count', 'sum', 'average'],
                'context': ['category', 'group']
            },
            'scatter': {
                'primary': ['scatter', 'scatter plot', 'correlation', 'relationship', 'vs', 'versus', 'x vs y', 'xy'],
                'secondary': ['correlate', 'covariance', 'regression'],
                'context': ['numeric', 'bivariate']
            },
            'line': {
                'primary': ['line', 'line chart', 'trend', 'over time', 'time series', 'timeline'],
                'secondary': ['date', 'month', 'year', 'period'],
                'context': ['time', 'date']
            },
            'histogram': {
                'primary': ['histogram', 'distribution', 'frequency', 'bins', 'density'],
                'secondary': ['continuous', 'numeric'],
                'context': ['distribution']
            },
            'box': {
                'primary': ['box', 'box plot', 'boxplot', 'quartile', 'outlier', 'median'],
                'secondary': ['statistics', 'summary'],
                'context': ['statistical']
            },
            'heatmap': {
                'primary': ['heatmap', 'correlation matrix', 'matrix plot', 'density map'],
                'secondary': ['correlations', 'relationships'],
                'context': ['matrix', 'multiple']
            },
            'area': {
                'primary': ['area', 'area chart', 'stacked area', 'cumulative', 'filled chart'],
                'secondary': ['cumulative', 'total over time'],
                'context': ['time']
            }
        }

        self.aggregation_terms = {
            'sum': ['sum', 'total', 'add up', 'aggregate', 'cumulative'],
            'mean': ['mean', 'average', 'avg'],
            'count': ['count', 'number', 'how many', 'frequency'],
            'max': ['max', 'maximum', 'highest'],
            'min': ['min', 'minimum', 'lowest'],
            'median': ['median']
        }

        # load the cached vectorizer/docs
        self._res = get_detector_resources()
        self.vectorizer = self._res['vectorizer']
        self.tfidf_matrix = self._res['tfidf_matrix']
        self.pattern_keys = self._res['pattern_keys']

    def _keyword_score(self, query):
        query_lower = query.lower()
        scores = {chart: 0.0 for chart in self.chart_patterns.keys()}
        for chart_type, patterns in self.chart_patterns.items():
            for kw in patterns.get('primary', []):
                if re.search(r'\b' + re.escape(kw.lower()) + r'\b', query_lower):
                    scores[chart_type] += 4.0
            for kw in patterns.get('secondary', []):
                if re.search(r'\b' + re.escape(kw.lower()) + r'\b', query_lower):
                    scores[chart_type] += 2.0
            for kw in patterns.get('context', []):
                if re.search(r'\b' + re.escape(kw.lower()) + r'\b', query_lower):
                    scores[chart_type] += 1.0
        # some heuristics
        if ' vs ' in query.lower() or ' versus ' in query.lower():
            scores['scatter'] += 5.0
            scores['bar'] += 2.0
        if any(x in query.lower() for x in ['over time', 'trend', 'timeline', 'time series']):
            scores['line'] += 5.0
            scores['area'] += 2.0
        return scores

    def _tfidf_score(self, query):
        if self.vectorizer is None or self.tfidf_matrix is None:
            return {k: 0.0 for k in self.pattern_keys}
        qvec = self.vectorizer.transform([query])
        sims = cosine_similarity(qvec, self.tfidf_matrix).flatten()
        return {self.pattern_keys[i]: float(sims[i]) for i in range(len(self.pattern_keys))}

    def detect_chart_type(self, query, df=None):
        if not query:
            return 'bar'
        kw = self._keyword_score(query)
        tf = self._tfidf_score(query)
        # scale tfidf small to bring into weight with keyword scores
        tf_scaled = {k: v * 3.0 for k, v in tf.items()}
        combined = {k: kw.get(k, 0.0) + tf_scaled.get(k, 0.0) for k in kw.keys()}
        # dataset aware adjustments (bias)
        if df is not None:
            numeric_count = df.select_dtypes(include=[np.number]).shape[1]
            if numeric_count >= 6:
                combined['heatmap'] += 6.0
            if numeric_count >= 2:
                combined['scatter'] += 1.5
        best = max(combined, key=combined.get)
        return best if combined[best] > 0 else 'bar'

    def detect_aggregation(self, query):
        q = query.lower()
        for agg_type, terms in self.aggregation_terms.items():
            for t in terms:
                if re.search(r'\b' + re.escape(t) + r'\b', q):
                    return agg_type
        return None


# -------------------------
# MANUAL COLUMN SELECTION SYSTEM (unchanged)
# -------------------------
def create_manual_controls(df, current_spec=None):
    """Create manual column selection controls"""
    st.markdown("### 🎯 Manual Column Selection")
    all_columns = df.columns.tolist()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    current_x = current_spec.get('x', all_columns[0] if all_columns else None)
    current_y = current_spec.get('y', all_columns[1] if len(all_columns) > 1 else None)
    current_chart = current_spec.get('chart_type', 'bar')
    current_agg = current_spec.get('aggregation', None)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("X-Axis Column")
        x_column = st.selectbox(
            "Select X-axis column:",
            options=all_columns,
            index=all_columns.index(current_x) if current_x in all_columns else 0,
            key="manual_x",
            help="Choose the column for X-axis (usually categories, time, or independent variable)"
        )
        if x_column:
            dtype = df[x_column].dtype
            unique_count = df[x_column].nunique()
            st.caption(f"Type: {dtype} | Unique values: {unique_count}")

    with col2:
        st.subheader("Y-Axis Column")
        y_column = st.selectbox(
            "Select Y-axis column:",
            options=['None'] + all_columns,
            index=0 if not current_y else all_columns.index(current_y) + 1,
            key="manual_y",
            help="Choose the column for Y-axis (usually numeric values or dependent variable)"
        )
        if y_column != 'None':
            dtype = df[y_column].dtype
            st.caption(f"Type: {dtype}")

    with col3:
        st.subheader("Chart Configuration")
        manual_chart_type = st.selectbox(
            "Chart Type:",
            options=['auto', 'bar', 'line', 'scatter', 'pie', 'histogram', 'box'],
            index=['auto', 'bar', 'line', 'scatter', 'pie', 'histogram', 'box'].index(current_chart) if current_chart in ['auto', 'bar', 'line', 'scatter', 'pie', 'histogram', 'box'] else 0,
            key="manual_chart_type"
        )
        aggregation = st.selectbox(
            "Aggregation:",
            options=['auto', 'sum', 'mean', 'count', 'max', 'min', 'median'],
            index=0 if not current_agg else ['auto', 'sum', 'mean', 'count', 'max', 'min', 'median'].index(current_agg),
            key="manual_aggregation"
        )

    group_column = st.selectbox(
        "Group By (optional - for colored charts):",
        options=['None'] + categorical_cols,
        index=0,
        key="manual_group",
        help="Choose a categorical column to group/color the data"
    )

    return {
        'x': x_column,
        'y': y_column if y_column != 'None' else None,
        'chart_type': manual_chart_type if manual_chart_type != 'auto' else None,
        'aggregation': aggregation if aggregation != 'auto' else None,
        'group': group_column if group_column != 'None' else None,
        'manual_mode': True
    }


# -------------------------
# ULTIMATE CHART GENERATOR (keeps your logic)
# -------------------------
def generate_ultimate_chart(df, user_query, manual_spec=None):
    """Generate charts with ultimate accuracy and manual controls"""
    if manual_spec and manual_spec.get('manual_mode'):
        x_col = manual_spec['x']
        y_col = manual_spec['y']
        chart_type = manual_spec.get('chart_type') or 'bar'
        aggregation = manual_spec.get('aggregation')
        hue_col = manual_spec.get('group')
        if not manual_spec.get('chart_type'):
            detector = UltimateChartDetector()
            chart_type = detector.detect_chart_type(user_query or "manual")
    else:
        detector = UltimateChartDetector()
        chart_type = detector.detect_chart_type(user_query, df=df)
        aggregation = detector.detect_aggregation(user_query)

        all_columns = df.columns.tolist()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

        x_col = all_columns[0] if all_columns else None
        y_col = all_columns[1] if len(all_columns) > 1 else None
        hue_col = None

        if chart_type == 'pie' and categorical_cols:
            x_col = categorical_cols[0]
        elif chart_type == 'histogram' and numeric_cols:
            x_col = numeric_cols[0]
        elif chart_type == 'scatter' and len(numeric_cols) >= 2:
            x_col, y_col = numeric_cols[0], numeric_cols[1]

    fig, ax = plt.subplots(figsize=(12, 7))

    try:
        plot_df = df.copy()
        if aggregation and x_col and y_col:
            if aggregation == 'sum':
                plot_data = df.groupby(x_col)[y_col].sum().reset_index()
            elif aggregation == 'mean':
                plot_data = df.groupby(x_col)[y_col].mean().reset_index()
            elif aggregation == 'count':
                plot_data = df.groupby(x_col)[y_col].count().reset_index()
            elif aggregation == 'max':
                plot_data = df.groupby(x_col)[y_col].max().reset_index()
            elif aggregation == 'min':
                plot_data = df.groupby(x_col)[y_col].min().reset_index()
            elif aggregation == 'median':
                plot_data = df.groupby(x_col)[y_col].median().reset_index()
            else:
                plot_data = df[[x_col, y_col]].copy()
        else:
            plot_data = df.copy()

        if chart_type == 'pie':
            col_to_plot = x_col
            value_counts = df[col_to_plot].value_counts().head(8)
            colors = plt.cm.Set3(np.linspace(0, 1, len(value_counts)))
            wedges, texts, autotexts = ax.pie(
                value_counts.values,
                labels=value_counts.index,
                autopct='%1.1f%%',
                startangle=90,
                colors=colors,
                shadow=True
            )
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
            ax.set_title(f'Pie Chart: {user_query}', fontsize=14, fontweight='bold')
            ax.axis('equal')

        elif chart_type == 'bar':
            if y_col:
                if aggregation and x_col in plot_data.columns and y_col in plot_data.columns:
                    bar_data = plot_data.set_index(x_col)[y_col]
                else:
                    bar_data = df.groupby(x_col)[y_col].mean()
                bars = ax.bar(range(len(bar_data)), bar_data.values,
                              color='skyblue', edgecolor='navy', alpha=0.8)
                ax.set_xticks(range(len(bar_data)))
                ax.set_xticklabels(bar_data.index, rotation=45, ha='right')
                ax.set_ylabel(y_col)
                ax.set_xlabel(x_col)
            else:
                value_counts = df[x_col].value_counts().head(15)
                bars = ax.bar(range(len(value_counts)), value_counts.values,
                              color='lightcoral', edgecolor='darkred', alpha=0.8)
                ax.set_xticks(range(len(value_counts)))
                ax.set_xticklabels(value_counts.index, rotation=45, ha='right')
                ax.set_ylabel('Count')
                ax.set_xlabel(x_col)
            ax.set_title(f'Bar Chart: {user_query}', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')

        elif chart_type == 'scatter':
            if x_col and y_col:
                ax.scatter(df[x_col], df[y_col], alpha=0.7, s=60, c='blue', edgecolors='black', linewidth=0.5)
                ax.set_xlabel(x_col, fontweight='bold')
                ax.set_ylabel(y_col, fontweight='bold')
                ax.set_title(f'Scatter Plot: {user_query}', fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3)

        elif chart_type == 'line':
            if x_col and y_col:
                # try date conversion for x-axis if possible
                try:
                    x_vals = pd.to_datetime(df[x_col], errors='coerce')
                    ax.plot(x_vals, df[y_col], marker='o', linewidth=2, color='red', markersize=4, alpha=0.8)
                    ax.xaxis.set_tick_params(rotation=45)
                except Exception:
                    ax.plot(df[x_col], df[y_col], marker='o', linewidth=2, color='red', markersize=4, alpha=0.8)
                ax.set_xlabel(x_col, fontweight='bold')
                ax.set_ylabel(y_col, fontweight='bold')
                ax.set_title(f'Line Chart: {user_query}', fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3)

        elif chart_type == 'histogram':
            col_to_plot = x_col or y_col
            ax.hist(df[col_to_plot].dropna(), bins=20, alpha=0.7, color='purple', edgecolor='black')
            ax.set_xlabel(col_to_plot, fontweight='bold')
            ax.set_ylabel('Frequency', fontweight='bold')
            ax.set_title(f'Histogram: {user_query}', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)

        elif chart_type == 'box':
            if y_col and y_col in df.columns:
                sns.boxplot(y=df[y_col], ax=ax)
                ax.set_title(f'Box Plot: {user_query}', fontsize=14, fontweight='bold')
            else:
                ax.text(0.5, 0.5, "Insufficient numeric column for boxplot", ha='center', va='center')

        elif chart_type == 'heatmap':
            num_df = df.select_dtypes(include=[np.number])
            if num_df.shape[1] >= 2:
                corr = num_df.corr()
                sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5, ax=ax)
                ax.set_title(f'Heatmap: Correlation Matrix\n{user_query}', fontsize=14, fontweight='bold')
            else:
                ax.text(0.5, 0.5, "Not enough numeric columns for heatmap", ha='center', va='center')

        elif chart_type == 'area':
            if x_col and y_col:
                try:
                    x_vals = pd.to_datetime(df[x_col], errors='coerce')
                    ax.fill_between(x_vals, df[y_col], alpha=0.5)
                    ax.plot(x_vals, df[y_col], marker='o')
                    ax.set_title(f'Area Chart: {user_query}', fontsize=14, fontweight='bold')
                    ax.set_xlabel(x_col); ax.set_ylabel(y_col)
                except Exception:
                    ax.text(0.5, 0.5, "Area chart requires a time-like x column", ha='center', va='center')

        plt.tight_layout()
        return fig, chart_type, {
            'x': x_col,
            'y': y_col,
            'hue': hue_col,
            'aggregation': aggregation,
            'manual_mode': manual_spec is not None
        }

    except Exception as e:
        ax.clear()
        ax.text(0.5, 0.5, f'Chart Generation Error\n\n{str(e)}', ha='center', va='center', transform=ax.transAxes, fontsize=11)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        return fig, 'error', {'x': None, 'y': None, 'hue': None, 'aggregation': None}


# -------------------------
# MAIN APPLICATION
# -------------------------
def main():
    # File upload - now supports csv, xls, xlsx, pdf
    st.sidebar.header("📁 Upload Your Dataset")
    uploaded_file = st.sidebar.file_uploader("Choose a file (CSV, XLSX, XLS, PDF)", type=['csv', 'xlsx', 'xls', 'pdf'])

    if uploaded_file is not None:
        try:
            # Use cached loader for performance
            file_bytes = uploaded_file.read()
            df = load_file_bytes(file_bytes, uploaded_file.name)
            # normalize columns to strings
            df.columns = [str(c) for c in df.columns]
            st.session_state.df = df
            st.sidebar.success("✅ Dataset loaded!")

            # Data overview
            st.sidebar.subheader("🔍 Data Overview")
            st.sidebar.write(f"**Records:** {df.shape[0]:,}")
            st.sidebar.write(f"**Columns:** {df.shape[1]}")

            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

            st.sidebar.write(f"**Numeric:** {len(numeric_cols)}")
            st.sidebar.write(f"**Categorical:** {len(categorical_cols)}")

            with st.sidebar.expander("📋 Data Quality Report"):
                st.write("**Column Summary:**")
                for col in df.columns:
                    dtype = df[col].dtype
                    unique_count = df[col].nunique()
                    null_count = df[col].isnull().sum()
                    st.write(f"• **{col}**: {dtype} | {unique_count} unique | {null_count} nulls")

        except Exception as e:
            st.sidebar.error(f"❌ Error: {str(e)}")
            return

    # Main content layout
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.header("🎯 Visualization Request")

        if st.session_state.df is not None:
            st.success("🚀 **Ready for ultra-accurate visualizations!**")

            # Mode toggle
            mode_cols = st.columns(2)
            with mode_cols[0]:
                if st.button("🤖 Automatic Mode", use_container_width=True):
                    st.session_state.manual_mode = False
            with mode_cols[1]:
                if st.button("👤 Manual Mode", use_container_width=True):
                    st.session_state.manual_mode = True

            if st.session_state.manual_mode:
                st.info("🔧 **Manual Mode Active** - Select columns manually for perfect control")
                current_spec = st.session_state.get('used_columns', {})
                manual_spec = create_manual_controls(st.session_state.df, current_spec)
                if st.button("🎯 Generate with Manual Selection", type="primary", use_container_width=True):
                    with st.spinner("Creating manual visualization..."):
                        fig, chart_type, used_columns = generate_ultimate_chart(st.session_state.df, "Manual Selection", manual_spec)
                        st.session_state.current_fig = fig
                        st.session_state.chart_generated = True
                        st.session_state.detected_chart_type = chart_type
                        st.session_state.used_columns = used_columns
                        st.session_state.manual_spec = manual_spec
            else:
                st.info("🤖 **Automatic Mode** - Describe what you want and AI will figure it out")
                user_query = st.text_area(
                    "Describe your visualization:",
                    value=st.session_state.get('user_query', ''),
                    height=100,
                    placeholder="Examples:\n• 'Pie chart showing category distribution'\n• 'Bar chart comparing sales across regions'\n• 'Scatter plot of price vs demand'\n• 'Line chart showing revenue trend over time'\n• 'Histogram of age distribution'\n",
                    help="Our improved detector uses TF-IDF + keywords for faster & better detection!"
                )

                if st.button("🚀 Generate Automatic Visualization", type="primary", use_container_width=True):
                    if user_query:
                        with st.spinner("🔍 Advanced AI analysis in progress..."):
                            fig, chart_type, used_columns = generate_ultimate_chart(st.session_state.df, user_query)
                            st.session_state.current_fig = fig
                            st.session_state.chart_generated = True
                            st.session_state.last_query = user_query
                            st.session_state.detected_chart_type = chart_type
                            st.session_state.used_columns = used_columns
                    else:
                        st.warning("Please enter a description")
            # --- LEFT BOTTOM: dataset preview (user requested "leftside bottom") ---
            st.markdown("---")
            st.markdown("### 🗂 Dataset Preview (left side bottom)")
            # show head in a scrollable container
            try:
                st.dataframe(st.session_state.df.head(100), use_container_width=True)
            except Exception:
                st.write(st.session_state.df.head())
        else:
            st.info("📤 Upload a CSV/XLSX/XLS/PDF file to get started!")

    with col_right:
        st.header("📈 Generated Visualization")

        if st.session_state.get('chart_generated', False) and st.session_state.get('current_fig'):
            chart_type = st.session_state.get('detected_chart_type', 'chart')
            used_columns = st.session_state.get('used_columns', {})

            if used_columns.get('manual_mode'):
                st.success("✅ **Manual Mode Visualization**")
            else:
                st.success("✅ **AI-Generated Visualization**")

            with st.expander("🤖 Analysis Report", expanded=True):
                c1, c2, c3 = st.columns(3)
                c1.metric("Chart Type", chart_type.upper())
                if used_columns.get('x'):
                    c2.metric("X-Axis", used_columns['x'])
                if used_columns.get('y'):
                    c3.metric("Y-Axis", used_columns['y'])
                if not used_columns.get('manual_mode') and st.session_state.get('last_query'):
                    st.write(f"**Your Request**: '{st.session_state.last_query}'")

            st.pyplot(st.session_state.current_fig)

            # quick switches
            s1, s2 = st.columns(2)
            with s1:
                if st.button("🔄 Switch to Manual", use_container_width=True):
                    st.session_state.manual_mode = True
                    st.experimental_rerun()
            with s2:
                if st.button("🔄 Switch to Auto", use_container_width=True):
                    st.session_state.manual_mode = False
                    st.experimental_rerun()
        else:
            st.markdown("""
            ### 🎯 Dual-Mode Visualization
            **🤖 Automatic Mode:** Describe in plain English — AI handles the rest.
            **👤 Manual Mode:** Select exact columns & aggregations for precise control.
            """)
            # Show a helpful hint with small preview if dataset exists
            if st.session_state.df is not None:
                st.markdown("**Hint:** After uploading, use Automatic mode to generate a chart quickly, then switch to Manual mode to fine-tune columns.")

    # Footer / features
    st.markdown("---")
    st.markdown("""
    **🎯 Improvements in this version:**
    - Multi-file support: CSV / Excel (XLSX/XLS) / basic PDF table extraction (pdfplumber)
    - Left-side dataset preview (shows top rows) after upload
    - Cached TF-IDF + pattern docs for faster detector initialization
    - Enriched semantic examples per chart type so detector handles diverse phrasing better
    - Cached file loading for snappy re-open of same dataset
    """)

if __name__ == "__main__":
    main()
