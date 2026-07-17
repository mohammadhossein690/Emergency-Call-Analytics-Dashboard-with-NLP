# ==========================================
# 911 Calls Dashboard 
# ==========================================

# Environment Setup & Configuration
import streamlit as st  # type: ignore
# Data Analysis Libraries
import numpy as np
import pandas as pd
# Visualization Libraries
import seaborn as sns  # type: ignore
import matplotlib.pyplot as plt
import plotly.express as px
#machine learning Libraries
import re
import nltk # type: ignore
from pathlib import Path
from nltk.corpus import stopwords # type: ignore
from nltk.stem.porter import PorterStemmer # type: ignore
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (confusion_matrix,accuracy_score,classification_report,
                            precision_score,recall_score,f1_score)

DATA_PATH = Path(__file__).parent / "911_cleaned.csv"

try:
    ENGLISH_STOP_WORDS = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords", quiet=True)
    ENGLISH_STOP_WORDS = set(stopwords.words("english"))


def clean_emergency_text(text, stemmer, stop_words):
    text = str(text)
    text = text.split(":", 1)[1].strip() if ":" in text else text.strip()
    text = re.sub(r"[^a-zA-Z]", " ", text)
    tokens = text.lower().split()
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

# This is page configuration
st.set_page_config(
    page_title="911 Calls Dashboard",
    page_icon="🚨",layout="wide")

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.7rem;
        padding-bottom: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

DAY_ORDER = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
MONTH_ORDER = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["timeStamp"])
    if "Date" not in df.columns:
        df["Date"] = df["timeStamp"].dt.date
    if "Reason" not in df.columns and "title" in df.columns:
        df["Reason"] = df["title"].apply(lambda x: x.split(":")[0])
    df["Day of Week"] = df["timeStamp"].dt.day_name().str[:3]
    df["Hour"] = df["timeStamp"].dt.hour
    df["Month"] = df["timeStamp"].dt.month_name().str[:3]
    df["Day of Week"] = pd.Categorical(df["Day of Week"], categories=DAY_ORDER, ordered=True)
    df["Month"] = pd.Categorical(df["Month"], categories=MONTH_ORDER, ordered=True)
    return df

# Load the dataset
df = load_data()

@st.cache_data
def get_reason_trend_data(reason_name):
    return (
        df[df["Reason"] == reason_name]
        .groupby("Date")
        .size()
        .reset_index(name="Calls")
    )

def plot_reason_trend(reason_name):
    trend_data = get_reason_trend_data(reason_name)
    fig = px.line(
        trend_data,x="Date",
        y="Calls",template="plotly_dark",height=400
    )
    fig.update_traces(line=dict(color=COLOR_MAP.get(reason_name, "#00a180"), width=2))
    fig.update_layout(
        xaxis_title="Date",yaxis_title="Number of Calls",
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig

COLOR_MAP = {
    "Fire": "#ef4444","EMS": "#3b82f6","Traffic": "#8b5cf6"
}

with st.sidebar:
    st.title("🚨 911 Project")
    main_page = st.radio(
        "Main Navigation",
        ["EDA Explorer", "Prediction Model", "Model Deep Dive", "Data Table"]
    )

@st.cache_resource(show_spinner=False)
def train_nlp_model():
    if "df" in globals():
        dataset = df
    else:
        dataset = pd.read_csv(DATA_PATH)

    required_cols = {"title", "Reason"}
    missing_cols = required_cols - set(dataset.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    dataset = dataset[["title", "Reason"]].dropna()

    stemmer = PorterStemmer()
    stop_words = ENGLISH_STOP_WORDS.copy()
    stop_words.update(["ems", "fire", "traffic"])


    dataset["processed_text"] = dataset["title"].apply(
        lambda x: clean_emergency_text(x, stemmer, stop_words)
    )

    dataset = dataset[dataset["processed_text"].str.strip() != ""].copy()

    X_text = dataset["processed_text"]
    y = dataset["Reason"]

    vectorizer = CountVectorizer(max_features=5000,min_df=2)
    X = vectorizer.fit_transform(X_text)

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X,
        y,
        dataset.index,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    classifier = MultinomialNB(alpha=0.001, fit_prior=False)
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)
    y_proba = classifier.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    classes = classifier.classes_
    cm = confusion_matrix(y_test, y_pred, labels=classes)

    report_dict = classification_report(
        y_test,
        y_pred,
        labels=classes,
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report_dict).transpose()

    cv_scores = [acc]

    best_params = {
    "alpha": 0.001,
    "fit_prior": False
    }
    best_score = acc

    feature_names = np.array(vectorizer.get_feature_names_out())
    top_keywords = {}

    for i, cls in enumerate(classes):
        top_idx = classifier.feature_log_prob_[i].argsort()[-10:][::-1]
        top_keywords[cls] = feature_names[top_idx].tolist()

    test_results_df = dataset.loc[idx_test, ["title", "Reason"]].copy()
    test_results_df["Predicted"] = y_pred
    test_results_df["Correct"] = test_results_df["Reason"] == test_results_df["Predicted"]
    test_results_df["Confidence"] = y_proba.max(axis=1)

    mistakes_df = test_results_df[~test_results_df["Correct"]].copy()
    mistakes_df = mistakes_df.sort_values("Confidence", ascending=False)

    class_distribution = (
        dataset["Reason"]
        .value_counts()
        .rename_axis("Reason")
        .reset_index(name="Count")
    )

    return {
        "vectorizer": vectorizer,
        "classifier": classifier,
        "classes": classes,
        "cm": cm,
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "report_df": report_df,
        "cv_mean": np.mean(cv_scores),
        "cv_std": np.std(cv_scores),
        "best_score": best_score,
        "best_params": best_params,
        "test_results_df": test_results_df,
        "mistakes_df": mistakes_df,
        "top_keywords": top_keywords,
        "class_distribution": class_distribution,
        "stemmer": stemmer,
        "stop_words": stop_words,
    }

if main_page == "EDA Explorer":
    st.markdown("## Exploratory Data Analysis: 911 Emergency Calls")
    eda_tab1, eda_tab2, eda_tab3, eda_tab4, eda_tab5 = st.tabs([
        "📈 Trends by Type","📊 Reason Distribution",
        "🔥 Pattern Heatmaps","📅 Weekly Patterns","Trend Estimation"])

    with eda_tab1:
        trend_view = st.radio(
            "Trend Perspective:",["Total Daily Volume", "By Emergency Type"],
            horizontal=True)

        if trend_view == "Total Daily Volume":
            st.markdown("#### Daily 911 Call Volume")
            daily_calls = df.groupby("Date").size().reset_index(name="Number of Calls")

            fig_daily_total = px.line(
                daily_calls,x="Date",y="Number of Calls",
                template="plotly_dark",height=400)
            fig_daily_total.update_traces(line=dict(color="#00F0FF", width=2))
            fig_daily_total.update_layout(
                xaxis_title="Date",yaxis_title="Number of Calls",
                margin=dict(l=20, r=20, t=40, b=20))

            st.plotly_chart(fig_daily_total, use_container_width=True)

        else:
            st.markdown("#### Daily Call Trends by Emergency Type")
            tab1, tab2, tab3 = st.tabs(["Traffic", "Fire", "EMS"])
            with tab1:
                st.plotly_chart(plot_reason_trend("Traffic"), use_container_width=True)
            with tab2:
                st.plotly_chart(plot_reason_trend("Fire"), use_container_width=True)
            with tab3:
                st.plotly_chart(plot_reason_trend("EMS"), use_container_width=True)

    with eda_tab2:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("#### Call Distribution by Reason")
            reason_counts = df["Reason"].value_counts()
            COLOR_MAP = {
                "EMS": "#fe2f3b","Fire": "#001be0","Traffic": "#8b5cf6"   }
            fig_pie = px.pie(
                names=reason_counts.index,values=reason_counts.values,
                color=reason_counts.index,color_discrete_map=COLOR_MAP,hole=0.4)
            fig_pie.update_traces(
                textposition="inside",textinfo="percent+label",
                marker=dict(line=dict(color="#111111", width=2)))

            fig_pie.update_layout(
                template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",height=400,
                margin=dict(l=20, r=20, t=20, b=20),showlegend=True,
                legend_title_text="Reason",font=dict(color="white"))
            st.plotly_chart(fig_pie, use_container_width=True)

        with col2:
            st.markdown("<h4 style='text-align: center;'>Geographic Hotspots</h4>", unsafe_allow_html=True)
            tab_zip, tab_twp = st.tabs(["Top 10 Zipcodes", "Top 10 Townships"])
            with tab_zip:
                top_zips = df["zip"].value_counts().head(10).reset_index()
                top_zips.columns = ["Zipcode", "Calls"]
                st.dataframe(top_zips, use_container_width=True, hide_index=True)
            with tab_twp:
                top_twps = df["twp"].value_counts().head(10).reset_index()
                top_twps.columns = ["Township", "Calls"]
                st.dataframe(top_twps, use_container_width=True, hide_index=True)

    with eda_tab3:
        custom_bg = "#777991"
        heatmap_view = st.radio(
            "Heatmap View:",["Day vs Hour", "Day vs Month"],horizontal=True
        )
        # Make sure these columns are clean and ordered
        df["Day of Week"] = pd.Categorical(df["Day of Week"], categories=DAY_ORDER, ordered=True)
        df["Month"] = df["timeStamp"].dt.month_name().str[:3]
        df["Month"] = pd.Categorical(df["Month"], categories=MONTH_ORDER, ordered=True)
        dayhour = df.pivot_table(
            index="Day of Week",columns="Hour",values="twp",aggfunc="count"
        ).reindex(DAY_ORDER)
        available_months = [m for m in MONTH_ORDER if df["Month"].astype(str).eq(m).any()]
        daymonth = (
            df.pivot_table(index="Day of Week", columns="Month", values="twp", aggfunc="count")
            .reindex(index=DAY_ORDER, columns=available_months)
            .fillna(0)
        )

        if heatmap_view == "Day vs Hour":
            st.markdown("##### Call Density by Day of Week and Hour")
            col1, col2 = st.columns([1.15, 1])
            with col1:
                fig_heatmap_hour = px.imshow(
                    dayhour,
                    labels=dict(x="Hour of Day", y="Day of Week", color="Number of Calls"),
                    x=dayhour.columns,y=dayhour.index,color_continuous_scale="Viridis",
                    template="plotly_dark",height=360)
                fig_heatmap_hour.update_layout(
                    xaxis=dict(tickmode="linear", tick0=0, dtick=1),
                    paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=20, r=20, t=20, b=20))
                st.plotly_chart(fig_heatmap_hour, use_container_width=True)

            with col2:
                plt.style.use("default")
                fig_cluster_hour = sns.clustermap(
                    dayhour.fillna(0),cmap="viridis",figsize=(6.2, 3.8),
                    dendrogram_ratio=0.15,cbar_kws={"label": "Calls"})
                fig_cluster_hour.fig.set_facecolor(custom_bg)
                fig_cluster_hour.ax_heatmap.set_facecolor(custom_bg)
                fig_cluster_hour.ax_row_dendrogram.set_facecolor(custom_bg)
                fig_cluster_hour.ax_col_dendrogram.set_facecolor(custom_bg)
                plt.setp(
                    fig_cluster_hour.ax_heatmap.get_xticklabels(),rotation=0,
                    fontsize=8,color="#E0E0E0")
                plt.setp(
                    fig_cluster_hour.ax_heatmap.get_yticklabels(),fontsize=8,
                    color="#E0E0E0")
                st.pyplot(fig_cluster_hour.fig, use_container_width=True)
                plt.close(fig_cluster_hour.fig)
            st.caption("Hover on the heatmap to inspect exact call volumes by weekday and hour.")

        else:
            st.markdown("##### Call Density by Day of Week and Month")
            col1, col2 = st.columns([1.15, 1])
            with col1:
                fig_heatmap_month = px.imshow(
                daymonth,
                labels=dict(x="Month", y="Day of Week", color="Number of Calls"),
                x=available_months,
                y=daymonth.index,
                color_continuous_scale="Viridis",
                template="plotly_dark",
                height=360,
            )

                fig_heatmap_month.update_layout(
                xaxis=dict(categoryorder="array", categoryarray=available_months),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=20, r=20, t=20, b=20),
            )

                st.plotly_chart(fig_heatmap_month, use_container_width=True)
            with col2:
                plt.style.use("default")
                fig_cluster_month = sns.clustermap(
                    daymonth.fillna(0),cmap="viridis",figsize=(6.2, 3.8),
                    dendrogram_ratio=0.15,cbar_kws={"label": "Calls"}
                )
                fig_cluster_month.fig.patch.set_facecolor(custom_bg)
                fig_cluster_month.ax_heatmap.set_facecolor(custom_bg)
                fig_cluster_month.ax_row_dendrogram.set_facecolor(custom_bg)
                fig_cluster_month.ax_col_dendrogram.set_facecolor(custom_bg)
                plt.setp(fig_cluster_month.ax_heatmap.get_xticklabels(), rotation=0, fontsize=8, color="black")
                plt.setp(fig_cluster_month.ax_heatmap.get_yticklabels(), fontsize=8, color="black")
                st.pyplot(fig_cluster_month.fig, use_container_width=True)
                plt.close(fig_cluster_month.fig)
            st.caption("This view compares weekday activity across months, including low-volume months.")

    with eda_tab4:
        st.markdown("### 📅 Weekly Emergency Call Patterns")
        fig_weekly = px.histogram(
            df,
            x="Day of Week",color="Reason",
            category_orders={"Day of Week": DAY_ORDER},
            color_discrete_map={"Traffic": "#E76F51","Fire": "#4C78A8","EMS": "#8E7DBE"},
            barmode="group",template="plotly_dark",height=400)
        fig_weekly.update_layout(
            xaxis_title="Day of Week",yaxis_title="Number of Calls",
            margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_weekly, use_container_width=True)
        st.info(
            """
            **Observation:** Sunday shows the lowest volume of traffic-related calls,
            likely due to reduced commuting. EMS calls remain relatively steady across the week.
            """
        )

    with eda_tab5:
        byMonth = (
            df.groupby("Month", observed=True)["twp"].count().reset_index())
        byMonth.columns = ["Month", "Number of Calls"]
        byMonth["Month_Index"] = range(1, len(byMonth) + 1)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Monthly Call Volume")
            fig_monthly_line = px.line(
                byMonth,x="Month",y="Number of Calls",
                markers=True,template="plotly_dark",height=550,)
            fig_monthly_line.update_traces(
                line=dict(color="#00a180", width=3),marker=dict(size=8, color="#00a180")
            )
            fig_monthly_line.update_layout(
                title="Trend of 911 Calls per Month",xaxis_title="Month",
                yaxis_title="Total Calls",paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",margin=dict(l=20, r=20, t=50, b=20),)
            st.plotly_chart(fig_monthly_line, use_container_width=True)
        with col2:
            st.markdown("#### Linear Trend Estimation")
            fig_trend = px.scatter(
                byMonth,
                x="Month_Index",y="Number of Calls",trendline="ols",
                trendline_color_override="#E76F51",
                labels={"Month_Index": "Month", "Number of Calls": "Total Calls"},
                template="plotly_dark",height=550,)
            fig_trend.update_layout(
                title="Monthly Calls with Linear Trend",
                xaxis=dict(tickmode="array",tickvals=byMonth["Month_Index"],
                    ticktext=byMonth["Month"],),xaxis_title="Month",
                yaxis_title="Total Calls",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=20, r=20, t=50, b=20),)
            fig_trend.update_traces(
                marker=dict(size=10, color="#00a180", line=dict(width=1, color="white")))
            st.plotly_chart(fig_trend, use_container_width=True)

# -------------------------------------------------------------------------------------------------
elif main_page == "Prediction Model":
    st.markdown("## Machine Learning Prediction")
    st.write("NLP-based classification of emergency call reasons using the `title` text.")

    try:
        results = train_nlp_model()
    except Exception as e:
        st.error(f"Model training failed: {e}")
        st.stop()

    tab1, tab2, tab3 = st.tabs(
        ["Overview", "Evaluation", "Live Prediction"]
    )

    with tab1:
        st.subheader("Model Overview")

        col1, col2, col3 = st.columns(3)
        col1.metric("Test Accuracy", f"{results['acc'] * 100:.2f}%")
        col2.metric("Weighted Precision", f"{results['precision'] * 100:.2f}%")
        col3.metric("Weighted Recall", f"{results['recall'] * 100:.2f}%")

        col4, col5, col6 = st.columns(3)
        col4.metric("Weighted F1-Score", f"{results['f1'] * 100:.2f}%")
        col5.metric("10-Fold CV Mean", f"{results['cv_mean'] * 100:.2f}%")
        col6.metric("CV Std", f"{results['cv_std'] * 100:.2f}%")

        st.markdown("### Best Hyperparameters")
        st.write(results["best_params"])

        st.info(
            "This model predicts `Reason` from the emergency call `title` using "
            "text cleaning, Bag-of-Words vectorization, and Multinomial Naive Bayes."
        )

    with tab2:
        st.subheader("Model Evaluation")

        col1, col2 = st.columns([1, 1.05])

        with col1:
            st.markdown("### Classification Report")
            st.dataframe(results["report_df"].round(4), use_container_width=True)

        with col2:
            st.markdown("### Confusion Matrix")

            cm_df = pd.DataFrame(
                results["cm"],
                index=results["classes"],
                columns=results["classes"],
            )

            fig_cm = px.imshow(
                cm_df,
                color_continuous_scale="Blues",
                aspect="auto",
            )

            fig_cm.update_traces(
                text=cm_df.values,
                texttemplate="%{text}",
            )

            fig_cm.update_layout(
                template="plotly_dark",
                height=380,
                margin=dict(t=0, b=20, l=20, r=20),
                coloraxis_showscale=False,
                xaxis_title="Predicted",
                yaxis_title="Actual",
            )
            st.plotly_chart(fig_cm, use_container_width=True)

    with tab3:
        st.subheader("Interactive Prediction")

        user_input = st.text_area(
            "Enter Emergency Text",
            placeholder="Example: vehicle accident with injuries",
        )

        if st.button("Predict Reason"):
            if user_input.strip():
                cleaned_text = clean_emergency_text(
                    user_input,
                    results["stemmer"],
                    results["stop_words"],
                )

                if cleaned_text.strip() == "":
                    st.warning(
                        "No valid tokens remained after preprocessing. Please enter a more descriptive text."
                    )
                else:
                    vectorized_text = results["vectorizer"].transform([cleaned_text])
                    prediction = results["classifier"].predict(vectorized_text)[0]

                    st.success(f"Predicted Reason: {prediction}")

            else:
                st.warning("Please enter a text description first.")

# -------------------------------------------------------------------------------------------------
elif main_page == "Model Deep Dive":

    st.markdown("## Model Deep Dive")
    st.caption("Compact technical view of the NLP classifier.")

    try:
        results = train_nlp_model()
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        st.stop()

    deep_tab1, deep_tab2, deep_tab3 = st.tabs(["Pipeline", "Insights", "Limits"])

    with deep_tab1:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Algorithm", "MultinomialNB")
        with col2:
            st.metric("Best Alpha", results["best_params"].get("alpha"))
        with col3:
            st.metric("Fit Prior", str(results["best_params"].get("fit_prior")))

        st.markdown("### Workflow")
        st.markdown(
            """
            - `Input` -> emergency call title
            - `Cleaning` -> lowercase, regex cleaning, stopword removal, stemming
            - `Vectorization` -> `CountVectorizer`
            - `Model` -> `MultinomialNB`
            - `Output` -> predicted class: `EMS`, `Fire`, `Traffic`
            """
        )

        st.info(
            "The model uses the processed `title` field as input and predicts the emergency "
            "reason category."
        )

    with deep_tab2:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Top Keywords")
            keyword_lines = []
            for cls in ["EMS", "Fire", "Traffic"]:
                words = ", ".join(results["top_keywords"].get(cls, [])[:8])
                keyword_lines.append(f"- `{cls}`: {words}")
            st.markdown("\n".join(keyword_lines))

        with col2:
            st.markdown("### Why It Performs Well")
            st.write(
                "The `title` field contains strong class-specific words, so a simple "
                "Bag-of-Words + Naive Bayes pipeline performs well."
            )

        st.markdown("### Misclassified Samples")
        if not results["mistakes_df"].empty:
            mistakes_df = results["mistakes_df"][
                ["title", "Reason", "Predicted", "Confidence"]
            ].head(3).copy()
            mistakes_df.columns = ["Title", "Actual", "Predicted", "Confidence"]
            mistakes_df["Confidence"] = mistakes_df["Confidence"].round(3)

            st.dataframe(
                mistakes_df,
                use_container_width=True,
                hide_index=True,
                height=140,
            )
        else:
            st.success("No misclassified samples found in the current test split.")

    with deep_tab3:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Current Limits")
            st.markdown(
                """
                - Text-only model
                - Bag-of-Words loses deeper context
                - Sensitive to unusual wording
                - No time or location features
                """
            )

        with col2:                              # ✅ هم‌سطح col1
            st.markdown("### Performance Snapshot")
            st.metric("Accuracy", f"{results['acc'] * 100:.2f}%")
            st.metric("F1-Score", f"{results['f1'] * 100:.2f}%")

        st.success(
            "Fast, simple, and effective for dashboard use; still improvable with richer text features and structured inputs."
        )

elif main_page == "Data Table":
    st.markdown("## Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Records", f"{df.shape[0]:,}")
    col2.metric("Titles", df["title"].nunique() if "title" in df.columns else "N/A")
    col3.metric("Townships", df["twp"].nunique() if "twp" in df.columns else "N/A")
    col4.metric("ZIP Codes", df["zip"].nunique())

    tab1, tab2 = st.tabs(["Overview", "Columns"])

    with tab1:
        left, right = st.columns([1.2, 1])

        with left:
            st.markdown("### Preview")
            preview_cols = ["title", "Reason", "twp", "zip", "timeStamp"]
            preview_cols = [col for col in preview_cols if col in df.columns]

            st.dataframe(
                df[preview_cols].head(3),
                use_container_width=True,
                hide_index=True,
                height=150,
            )

        with right:
            st.markdown("### Info")
            st.markdown(
                """
                - **Goal:** analyze 911 emergency call patterns
                - **Spatial:** lat, lng, township, address
                - **Temporal:** timestamp of calls
                - **Incident:** title, reason, zipcode
                """
            )

    with tab2:
        info_df = pd.DataFrame(
            {
                "Column": df.columns,
                "Non-Null": df.count().values,
                "Type": df.dtypes.astype(str).values,
            }
        )
        st.dataframe(
            info_df,
            use_container_width=True,
            hide_index=True,
            height=220,
        )
    