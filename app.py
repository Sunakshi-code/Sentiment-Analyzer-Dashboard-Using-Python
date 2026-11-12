
import streamlit as st
import pandas as pd
import plotly.express as px
import nltk
import re

from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from collections import Counter

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Sentiment Analyzer Dashboard",
    page_icon="📊",
    layout="wide"
)

# ==========================================
# 2. NLP RESOURCE SETUP
# ==========================================

@st.cache_resource
def load_nlp_resources():

    nltk.download("vader_lexicon", quiet=True)
    nltk.download("stopwords", quiet=True)

    sia = SentimentIntensityAnalyzer()
    stop_words = set(stopwords.words("english"))

    # Preserve important sentiment words
    stop_words -= {"not", "no", "nor", "never"}

    return sia, stop_words


try:
    sia, stop_words = load_nlp_resources()

except Exception as e:
    st.error(f"Unable to load NLP resources: {e}")
    st.stop()


# ==========================================
# 3. TEXT PREPROCESSING
# ==========================================

def clean_text(text):

    text = str(text).lower()

    # Remove special characters and numbers
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # Tokenization
    words = text.split()

    # Stopword removal
    words = [
        word for word in words
        if word not in stop_words
    ]

    return " ".join(words)


# ==========================================
# 4. SENTIMENT CLASSIFICATION
# ==========================================

def analyze_sentiment(text):

    score = sia.polarity_scores(str(text))["compound"]

    if score >= 0.05:
        return "Positive"

    elif score <= -0.05:
        return "Negative"

    else:
        return "Neutral"


# ==========================================
# 5. DATA CLEANING AND ANALYSIS
# ==========================================

def analyze_dataset(data):

    data = data.copy()

    # Remove missing reviews
    data = data.dropna(
        subset=["Customer_Review"]
    )

    # Convert reviews into strings
    data["Customer_Review"] = (
        data["Customer_Review"]
        .astype(str)
        .str.strip()
    )

    # Remove empty reviews
    data = data[
        data["Customer_Review"] != ""
    ]

    # Remove duplicate reviews
    data = data.drop_duplicates(
        subset=["Customer_Review"]
    )

    data = data.reset_index(drop=True)

    # Create Review ID if unavailable
    if "Review_ID" not in data.columns:
        data.insert(
            0,
            "Review_ID",
            range(1, len(data) + 1)
        )

    # Clean reviews
    data["Cleaned_Review"] = (
        data["Customer_Review"].apply(clean_text)
    )

    # Sentiment classification
    data["Sentiment"] = (
        data["Customer_Review"].apply(analyze_sentiment)
    )

    # Calculate compound score
    data["Sentiment_Score"] = data[
        "Customer_Review"
    ].apply(
        lambda text: sia.polarity_scores(text)["compound"]
    )

    return data


# ==========================================
# 6. DASHBOARD HEADER
# ==========================================

st.title("📊 Customer Sentiment Analyzer")

st.markdown(
    "### Understand customer opinions using NLP and Python"
)

st.caption(
    "Python | Pandas | NLTK | VADER | Streamlit | Plotly"
)

st.divider()


# ==========================================
# 7. SIDEBAR AND DATA UPLOAD
# ==========================================

st.sidebar.title("⚙️ Dashboard Settings")

st.sidebar.markdown("### Upload Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload Customer Reviews CSV",
    type=["csv"]
)


# ==========================================
# 8. LOAD DATASET
# ==========================================

if uploaded_file is not None:

    try:
        raw_df = pd.read_csv(uploaded_file)

    except Exception as e:
        st.error(f"Unable to read uploaded CSV: {e}")
        st.stop()

else:

    try:
        raw_df = pd.read_csv(
            "dataset/customer_reviews.csv"
        )

    except FileNotFoundError:
        st.error(
            "Dataset not found. Upload a CSV file "
            "or add dataset/customer_reviews.csv."
        )
        st.stop()

    except Exception as e:
        st.error(f"Dataset loading error: {e}")
        st.stop()


# Validate dataset
if "Customer_Review" not in raw_df.columns:

    st.error(
        "Required column missing: Customer_Review"
    )
    st.stop()


# Analyze uploaded or default dataset
df = analyze_dataset(raw_df)

if df.empty:
    st.warning("No valid customer reviews available.")
    st.stop()


st.sidebar.success("Dataset loaded successfully!")

st.sidebar.metric(
    "Total Valid Reviews",
    len(df)
)


# ==========================================
# 9. KPI SUMMARY
# ==========================================

positive = (df["Sentiment"] == "Positive").sum()

negative = (df["Sentiment"] == "Negative").sum()

neutral = (df["Sentiment"] == "Neutral").sum()

total_reviews = len(df)

positive_percent = positive / total_reviews * 100

negative_percent = negative / total_reviews * 100

neutral_percent = neutral / total_reviews * 100


st.subheader("📌 Overall Summary")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Reviews",
    total_reviews
)

col2.metric(
    "😊 Positive Reviews",
    positive
)

col3.metric(
    "😞 Negative Reviews",
    negative
)

col4.metric(
    "😐 Neutral Reviews",
    neutral
)


# ==========================================
# 10. SENTIMENT PERCENTAGES
# ==========================================

st.divider()

st.subheader("📊 Sentiment Percentage")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Positive",
    f"{positive_percent:.2f}%"
)

col2.metric(
    "Negative",
    f"{negative_percent:.2f}%"
)

col3.metric(
    "Neutral",
    f"{neutral_percent:.2f}%"
)


# ==========================================
# 11. SENTIMENT DISTRIBUTION CHARTS
# ==========================================

st.divider()

st.subheader("📈 Sentiment Analysis")

sentiment_data = pd.DataFrame({
    "Sentiment": [
        "Positive",
        "Negative",
        "Neutral"
    ],
    "Count": [
        positive,
        negative,
        neutral
    ]
})

chart1, chart2 = st.columns(2)


# Donut Chart
with chart1:

    fig_pie = px.pie(
        sentiment_data,
        names="Sentiment",
        values="Count",
        title="Sentiment Distribution",
        hole=0.45
    )

    st.plotly_chart(
        fig_pie,
        use_container_width=True
    )


# Bar Chart
with chart2:

    fig_bar = px.bar(
        sentiment_data,
        x="Sentiment",
        y="Count",
        color="Sentiment",
        text="Count",
        title="Sentiment Count Comparison"
    )

    st.plotly_chart(
        fig_bar,
        use_container_width=True
    )


# ==========================================
# 12. SENTIMENT SCORE ANALYSIS
# ==========================================

st.divider()

st.subheader("📉 Sentiment Score Distribution")

fig_hist = px.histogram(
    df,
    x="Sentiment_Score",
    color="Sentiment",
    nbins=20,
    title="Compound Sentiment Score Analysis"
)

st.plotly_chart(
    fig_hist,
    use_container_width=True
)


# ==========================================
# 13. MOST COMMON WORDS
# ==========================================

st.divider()

st.subheader("🔤 Most Frequently Used Words")

all_words = " ".join(
    df["Cleaned_Review"].astype(str)
).split()

word_counts = Counter(
    all_words
).most_common(10)

if word_counts:

    word_df = pd.DataFrame(
        word_counts,
        columns=["Word", "Frequency"]
    )

    fig_words = px.bar(
        word_df,
        x="Word",
        y="Frequency",
        text="Frequency",
        title="Top 10 Frequently Used Words"
    )

    st.plotly_chart(
        fig_words,
        use_container_width=True
    )

else:

    st.info("No words available for analysis.")


# ==========================================
# 14. CUSTOMER REVIEW FILTER
# ==========================================

st.divider()

st.subheader("🔍 Customer Reviews Analysis")

selected_sentiment = st.selectbox(
    "Filter Reviews by Sentiment",
    [
        "All",
        "Positive",
        "Negative",
        "Neutral"
    ]
)

if selected_sentiment == "All":

    filtered_df = df

else:

    filtered_df = df[
        df["Sentiment"] == selected_sentiment
    ]


st.dataframe(
    filtered_df,
    use_container_width=True,
    hide_index=True
)

st.caption(
    f"Displaying {len(filtered_df)} "
    f"out of {total_reviews} reviews"
)


# ==========================================
# 15. DOWNLOAD ANALYZED DATASET
# ==========================================

st.divider()

st.subheader("📥 Download Sentiment Report")

csv_data = df.to_csv(
    index=False
).encode("utf-8")

st.download_button(
    label="Download Analyzed CSV",
    data=csv_data,
    file_name="sentiment_analysis_report.csv",
    mime="text/csv"
)


# ==========================================
# 16. FOOTER
# ==========================================

st.divider()

st.caption(
    "Sentiment Analyzer Dashboard | Python & NLP"
)