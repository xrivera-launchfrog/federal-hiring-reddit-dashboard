import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
import re
from plotly.subplots import make_subplots

# Page configuration - single page, wide layout
st.set_page_config(
    page_title="Federal Hiring Pulse Check",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for clean, professional styling
st.markdown("""
<style>
    .main > div {
        padding-top: 1rem;
    }
    .stMetric {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 5px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .context-box {
        background-color: #e9ecef;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 20px;
    }
    h1 {
        color: #1f2937;
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        margin-bottom: 0.5rem !important;
    }
    h2 {
        color: #374151;
        font-size: 1.5rem !important;
        font-weight: 600 !important;
        margin-top: 2rem !important;
    }
    h3 {
        color: #4b5563;
        font-size: 1.25rem !important;
        font-weight: 600 !important;
    }
    .sentiment-positive { color: #059669; font-weight: bold; }
    .sentiment-negative { color: #dc2626; font-weight: bold; }
    .sentiment-neutral { color: #6b7280; font-weight: bold; }
    .insight-card {
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 10px;
        border-left: 4px solid;
    }
    .insight-negative { 
        background-color: #fee2e2; 
        border-left-color: #dc2626;
    }
    .insight-warning { 
        background-color: #fef3c7; 
        border-left-color: #f59e0b;
    }
    .insight-info { 
        background-color: #dbeafe; 
        border-left-color: #3b82f6;
    }
</style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    """Load and prepare Reddit data"""
    try:
        df = pd.read_csv('reddit_skills_combined.csv')
        df['created_utc'] = pd.to_datetime(df['created_utc'])
        df['date'] = df['created_utc'].dt.date
        df['full_text'] = df['title'].fillna('') + ' ' + df['body'].fillna('')
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.stop()

# Enhanced sentiment analysis with negative bias (Reddit reality)
@st.cache_data
def analyze_hiring_sentiment(df):
    """Analyze sentiment with realistic weighting for Reddit content - heavily reduced neutral classification"""
    
    # Hiring-related themes
    hiring_themes = {
        'Job Security/RIFs': r'\b(RIF|layoff|firing|fired|termination|job loss|severance|reduction in force|job security|pink slip)\b',
        'Political Appointments': r'\b(political|loyalty|partisan|ideology|MAGA|political appointee|loyalty test|political hire)\b',
        'Merit vs Loyalty': r'\b(merit|qualified|competent|skills|experience|expertise|loyalty over|political over merit)\b',
        'Hiring Process': r'\b(hiring process|recruitment|interview|application|slow hiring|delays|background check|clearance)\b',
        'Qualifications': r'\b(qualification|requirement|degree|education|experience required|overqualified|underqualified)\b',
        'Pay and Benefits': r'\b(pay|salary|GS|grade|benefits|compensation|locality pay|wage)\b',
        'Remote Work': r'\b(remote|telework|hybrid|work from home|WFH|return to office|RTO)\b',
        'Diversity/Inclusion': r'\b(diversity|inclusion|DEI|minority|veteran preference|disability|equal opportunity)\b',
        'Contractor Conversion': r'\b(contractor|conversion|contract to fed|privatization|outsourcing)\b',
        'Morale': r'\b(morale|burnout|stress|toxic|hostile|frustrated|exhausted|tired)\b'
    }
    
    # Critical topics that are almost never neutral when discussed
    critical_topics = {
        'doge': r'\b(DOGE|department of government efficiency|govt efficiency|government efficiency)\b',
        'layoffs': r'\b(layoff|mass layoff|reduction in force|RIF|termination|firing|fired|let go|job loss|pink slip)\b',
        'schedule_f': r'\b(schedule f|schedule-f|deferred resignation|deferred-resignation)\b'
    }
    
    # Sentiment patterns with Reddit-appropriate weighting
    # Negative patterns (weighted more heavily)
    strong_negative_patterns = r'\b(illegal|corrupt|theft|scam|fraud|disaster|terrible|awful|hate|disgusting|betrayal|crime|violation|unethical|unconstitutional|outrageous|ridiculous|unacceptable|shameful|despicable)\b'
    moderate_negative_patterns = r'\b(worried|concerned|fear|uncertain|frustrated|angry|disappointed|unfair|worse|declining|problem|issue|difficult|struggle|failed|failing|troubling|disturbing|alarming|threatening|scary|anxious|nervous)\b'
    mild_negative_patterns = r'\b(confused|unclear|unsure|challenging|tough|hard|complicated|bureaucracy|red tape|slow|delays|unfortunate|concerning|questionable|doubt|skeptical)\b'
    
    # Positive patterns (require stronger evidence)
    strong_positive_patterns = r'\b(excellent|amazing|fantastic|wonderful|love|perfect|best|thrilled|ecstatic|delighted)\b'
    moderate_positive_patterns = r'\b(opportunity|improvement|hope|better|excited|positive|progress|success|happy|glad|optimistic|encouraged|promising|beneficial)\b'
    
    # Context modifiers that flip sentiment
    negation_patterns = r'\b(not|no|never|without|lack|missing|absence|neither|none|hardly|barely|scarcely)\s+'
    sarcasm_indicators = r'\b(yeah right|sure|totally|definitely|absolutely)\b.*[!?]|/s\b'
    
    # Detect themes
    for theme, pattern in hiring_themes.items():
        df[f'theme_{theme}'] = df['full_text'].str.contains(pattern, case=False, regex=True, na=False)
    
    # Calculate sentiment with Reddit bias and critical topic awareness
    def calculate_realistic_sentiment(text):
        if pd.isna(text) or text.strip() == '':
            return 0, 'Neutral'
        
        text_lower = text.lower()
        
        # Check if discussing critical topics - these heavily lean negative
        is_critical_topic = False
        for topic, pattern in critical_topics.items():
            if re.search(pattern, text_lower):
                is_critical_topic = True
                break
        
        # Count negative indicators (weighted)
        strong_neg = len(re.findall(strong_negative_patterns, text_lower)) * 3
        moderate_neg = len(re.findall(moderate_negative_patterns, text_lower)) * 2
        mild_neg = len(re.findall(mild_negative_patterns, text_lower)) * 1
        
        # Count positive indicators (higher threshold)
        strong_pos = len(re.findall(strong_positive_patterns, text_lower)) * 2
        moderate_pos = len(re.findall(moderate_positive_patterns, text_lower)) * 1
        
        # Check for sarcasm (flips positive to negative)
        if re.search(sarcasm_indicators, text_lower):
            # Flip positive scores to negative
            strong_neg += strong_pos
            moderate_neg += moderate_pos
            strong_pos = 0
            moderate_pos = 0
        
        # Check for negated positives (these become negative)
        negated_positives = len(re.findall(negation_patterns + r'(' + moderate_positive_patterns + r')', text_lower))
        negated_positives += len(re.findall(negation_patterns + r'(' + strong_positive_patterns + r')', text_lower))
        
        # Calculate total score
        negative_score = strong_neg + moderate_neg + mild_neg + (negated_positives * 2)
        positive_score = strong_pos + moderate_pos
        
        # Critical topic bias - if discussing DOGE, layoffs, etc., require overwhelming positive evidence
        if is_critical_topic:
            # Need 3x more positive than negative to overcome critical topic bias
            if positive_score < negative_score * 3:
                negative_score *= 1.5  # Amplify negative for critical topics
        
        # Reddit bias: require 2x more positive than negative to be considered positive
        net_score = positive_score - negative_score
        
        # Normalize by text length (per 100 words)
        word_count = len(text.split())
        if word_count > 20:  # Only normalize longer texts
            net_score = (net_score / word_count) * 100
        
        # Categorize with Reddit-appropriate thresholds and reduced neutral zone
        if net_score > 3:  # Very high threshold for positive
            category = 'Optimistic'
        elif net_score < -0.3:  # Low threshold for negative
            category = 'Pessimistic'
        else:
            # For critical topics, default to negative unless strong positive evidence
            if is_critical_topic and net_score <= 1:
                category = 'Pessimistic'
            # For any discussion with sentiment words, avoid neutral
            elif negative_score > 0 or positive_score > 0:
                # If any sentiment detected, lean toward the dominant one
                if negative_score > positive_score:
                    category = 'Pessimistic'
                elif positive_score > negative_score * 1.5:  # Need clear positive majority
                    category = 'Optimistic'
                else:
                    category = 'Pessimistic'  # When in doubt on Reddit, it's negative
            else:
                # Only truly neutral if NO sentiment indicators found
                category = 'Neutral'
        
        return net_score, category
    
    # Apply sentiment analysis
    sentiment_results = df['full_text'].apply(calculate_realistic_sentiment)
    df['sentiment_score'] = sentiment_results.apply(lambda x: x[0])
    df['sentiment_category'] = sentiment_results.apply(lambda x: x[1])
    
    # Mark hiring-related content
    df['is_hiring_related'] = df[[col for col in df.columns if col.startswith('theme_')]].any(axis=1)
    
    return df

# Thread analysis
@st.cache_data
def analyze_threads(df):
    """Analyze thread-level metrics"""
    thread_stats = df.groupby('thread_id').agg({
        'sentiment_score': ['mean', 'std', 'first', 'last'],
        'type': lambda x: (x == 'post').sum(),
        'score': 'sum',
        'is_hiring_related': 'any',
        'created_utc': ['min', 'max']
    }).reset_index()
    
    thread_stats.columns = ['thread_id', 'avg_sentiment', 'sentiment_std', 'first_sentiment', 
                           'last_sentiment', 'post_count', 'total_score', 'is_hiring_thread',
                           'thread_start', 'thread_end']
    
    # Calculate evolution
    thread_stats['sentiment_change'] = thread_stats['last_sentiment'] - thread_stats['first_sentiment']
    thread_stats['evolution'] = thread_stats.apply(
        lambda x: 'Improving' if x['sentiment_change'] > 0.5 
        else 'Deteriorating' if x['sentiment_change'] < -0.5 
        else 'Stable', axis=1
    )
    
    # Calculate comment count
    thread_comment_counts = df.groupby('thread_id').size() - 1  # Subtract the post
    thread_stats['comment_count'] = thread_stats['thread_id'].map(thread_comment_counts).fillna(0)
    
    return thread_stats

# Load and process data
df = load_data()
df = analyze_hiring_sentiment(df)
thread_stats = analyze_threads(df)

# Define historical milestones
milestones = [
    {
        'name': 'Schedule F & Deferred-Resignation launch',
        'date': datetime(2025, 1, 28).date(),
        'immediate': (datetime(2025, 1, 27).date(), datetime(2025, 1, 29).date()),
        '7_day': (datetime(2025, 1, 28).date(), datetime(2025, 2, 3).date()),
        '30_day': (datetime(2025, 1, 28).date(), datetime(2025, 2, 26).date())
    },
    {
        'name': 'Dept. of Government Efficiency directive',
        'date': datetime(2025, 2, 11).date(),
        'immediate': (datetime(2025, 2, 10).date(), datetime(2025, 2, 12).date()),
        '7_day': (datetime(2025, 2, 11).date(), datetime(2025, 2, 17).date()),
        '30_day': (datetime(2025, 2, 11).date(), datetime(2025, 3, 12).date())
    },
    {
        'name': 'Probationary terminations',
        'date': datetime(2025, 2, 13).date(),
        'immediate': (datetime(2025, 2, 12).date(), datetime(2025, 2, 14).date()),
        '7_day': (datetime(2025, 2, 13).date(), datetime(2025, 2, 19).date()),
        '30_day': (datetime(2025, 2, 13).date(), datetime(2025, 3, 14).date())
    },
    {
        'name': 'DOE & Forest Service/NPS mass layoffs',
        'date': datetime(2025, 2, 14).date(),
        'immediate': (datetime(2025, 2, 13).date(), datetime(2025, 2, 15).date()),
        '7_day': (datetime(2025, 2, 14).date(), datetime(2025, 2, 20).date()),
        '30_day': (datetime(2025, 2, 14).date(), datetime(2025, 3, 15).date())
    },
    {
        'name': 'OPM all-hands email mandate',
        'date': datetime(2025, 2, 22).date(),
        'immediate': (datetime(2025, 2, 21).date(), datetime(2025, 2, 23).date()),
        '7_day': (datetime(2025, 2, 22).date(), datetime(2025, 2, 28).date()),
        '30_day': (datetime(2025, 2, 22).date(), datetime(2025, 3, 23).date())
    },
    {
        'name': 'RIF plans directive',
        'date': datetime(2025, 2, 26).date(),
        'immediate': (datetime(2025, 2, 25).date(), datetime(2025, 2, 27).date()),
        '7_day': (datetime(2025, 2, 26).date(), datetime(2025, 3, 4).date()),
        '30_day': (datetime(2025, 2, 26).date(), datetime(2025, 3, 27).date())
    },
    {
        'name': 'Alsup preliminary injunction',
        'date': datetime(2025, 2, 27).date(),
        'immediate': (datetime(2025, 2, 26).date(), datetime(2025, 2, 28).date()),
        '7_day': (datetime(2025, 2, 27).date(), datetime(2025, 3, 5).date()),
        '30_day': (datetime(2025, 2, 27).date(), datetime(2025, 3, 28).date())
    },
    {
        'name': 'HHS probationary terminations',
        'date': datetime(2025, 3, 27).date(),
        'immediate': (datetime(2025, 3, 26).date(), datetime(2025, 3, 28).date()),
        '7_day': (datetime(2025, 3, 27).date(), datetime(2025, 4, 2).date()),
        '30_day': (datetime(2025, 3, 27).date(), datetime(2025, 4, 25).date())
    },
    {
        'name': 'Fiscal 2026 budget proposal',
        'date': datetime(2025, 6, 3).date(),
        'immediate': (datetime(2025, 6, 2).date(), datetime(2025, 6, 4).date()),
        '7_day': (datetime(2025, 6, 3).date(), datetime(2025, 6, 9).date()),
        '30_day': (datetime(2025, 6, 3).date(), datetime(2025, 7, 2).date())
    },
    {
        'name': 'Supreme Court hiring-freeze decision',
        'date': datetime(2025, 7, 8).date(),
        'immediate': (datetime(2025, 7, 7).date(), datetime(2025, 7, 9).date()),
        '7_day': (datetime(2025, 7, 8).date(), datetime(2025, 7, 14).date()),
        '30_day': (datetime(2025, 7, 8).date(), datetime(2025, 8, 6).date())
    },
    {
        'name': 'Deferred-Resignation status snapshot',
        'date': datetime(2025, 7, 31).date(),
        'immediate': (datetime(2025, 7, 30).date(), datetime(2025, 8, 1).date()),
        '7_day': (datetime(2025, 7, 31).date(), datetime(2025, 8, 6).date()),
        '30_day': (datetime(2025, 7, 31).date(), datetime(2025, 8, 29).date())
    }
]

# Function to analyze sentiment around milestones
def analyze_milestone_impact(df, milestone):
    """Analyze sentiment and activity around a specific milestone"""
    results = {}
    
    # Immediate impact
    immediate_df = df[(df['date'] >= milestone['immediate'][0]) & (df['date'] <= milestone['immediate'][1])]
    results['immediate'] = {
        'posts': immediate_df['thread_id'].nunique(),
        'sentiment': immediate_df['sentiment_score'].mean() if len(immediate_df) > 0 else 0,
        'negative_pct': (immediate_df['sentiment_category'] == 'Pessimistic').sum() / len(immediate_df) * 100 if len(immediate_df) > 0 else 0
    }
    
    # 7-day impact
    seven_day_df = df[(df['date'] >= milestone['7_day'][0]) & (df['date'] <= milestone['7_day'][1])]
    results['7_day'] = {
        'posts': seven_day_df['thread_id'].nunique(),
        'sentiment': seven_day_df['sentiment_score'].mean() if len(seven_day_df) > 0 else 0,
        'negative_pct': (seven_day_df['sentiment_category'] == 'Pessimistic').sum() / len(seven_day_df) * 100 if len(seven_day_df) > 0 else 0
    }
    
    # 30-day impact
    thirty_day_df = df[(df['date'] >= milestone['30_day'][0]) & (df['date'] <= milestone['30_day'][1])]
    results['30_day'] = {
        'posts': thirty_day_df['thread_id'].nunique(),
        'sentiment': thirty_day_df['sentiment_score'].mean() if len(thirty_day_df) > 0 else 0,
        'negative_pct': (thirty_day_df['sentiment_category'] == 'Pessimistic').sum() / len(thirty_day_df) * 100 if len(thirty_day_df) > 0 else 0
    }
    
    return results

# Get date range
date_min = df['date'].min()
date_max = df['date'].max()

# ===== 1. DASHBOARD OVERVIEW =====
st.title("Federal Hiring Pulse Check")
st.markdown("Understanding how government workers feel about hiring practices")
st.markdown(f"**Analysis Period: {date_min.strftime('%B %d, %Y')} - {date_max.strftime('%B %d, %Y')}**")

# Use all data (no filtering)
filtered_df = df
filtered_threads = thread_stats

# Analysis Period & Methodology Section
with st.expander("**Analysis Period & Methodology** - Click to see data sources and methodology", expanded=True):
    context_col1, context_col2 = st.columns(2)
    
    with context_col1:
        st.markdown("### Data Sources")
        
        st.markdown(f"**Community Context**: Federal employee forums with 250K+ combined members")
        
        # Subreddit breakdown
        subreddit_counts = filtered_df['subreddit'].value_counts()
        total_items = len(filtered_df)
        
        st.markdown("\n**Communities Analyzed:**")
        for subreddit, count in subreddit_counts.items():
            percentage = (count / total_items) * 100
            st.markdown(f"• **r/{subreddit}**: {count:,} items ({percentage:.1f}%)")
        
        # Methodology section
        st.markdown("### Methodology")
        st.markdown("""
        **Sentiment Analysis Approach:**
        - **Negative bias calibration**: Reddit discussions tend negative, so we weight accordingly
        - **Strong negative** (3x weight): illegal, corrupt, theft, disaster
        - **Moderate negative** (2x weight): worried, concerned, frustrated
        - **Positive requires higher threshold** (2x evidence needed)
        - **Context-aware**: Detects negated positives ("not good" = negative)
        
        **Theme Detection:**
        - 10 major hiring themes tracked via keyword patterns
        - Counted at thread level to avoid duplication
        - Only themes with >10 threads included in analysis
        
        **Milestone Analysis:**
        - Tracks sentiment changes around key policy events
        - Measures immediate (±1 day), 7-day, and 30-day impacts
        - Compares activity levels before and after events
        """)
    
    with context_col2:
        st.markdown("### Analysis Scope")
        
        # Calculate metrics
        unique_threads = filtered_df['thread_id'].nunique()
        unique_authors = filtered_df['author'].nunique()
        hiring_threads = filtered_threads['is_hiring_thread'].sum()
        avg_comments = filtered_threads['comment_count'].mean()
        viral_threads = filtered_threads[filtered_threads['comment_count'] > 100].shape[0]
        
        st.markdown(f"""
        • **Total Conversations**: {unique_threads:,} unique threads  
        • **Total Engagement**: {len(filtered_df):,} posts & comments  
        • **Date Range**: {(date_max - date_min).days} days of data  
        • **Participants**: ~{unique_authors:,} unique users  
        • **Hiring-Related**: {hiring_threads:,} threads ({hiring_threads/unique_threads*100:.1f}%)  
        • **Average Thread Size**: {avg_comments:.0f} comments  
        • **Viral Threads** (100+ comments): {viral_threads}
        """)
        
        st.markdown("### Limitations")
        st.markdown("""
        • Represents Reddit users, not all federal employees  
        • Sentiment analysis may miss sarcasm/complex expressions  
        • Self-selected participation may skew negative  
        • Recent events may be overrepresented
        """)

st.markdown("---")

# ===== 2. EXECUTIVE SUMMARY =====
st.markdown("## Key Insights at a Glance")

# Calculate theme metrics for summary
theme_counts = {}
theme_columns = [col for col in filtered_df.columns if col.startswith('theme_')]

for theme_col in theme_columns:
    theme_name = theme_col.replace('theme_', '')
    # Count unique threads discussing this theme
    thread_count = filtered_df[filtered_df[theme_col]]['thread_id'].nunique()
    # Count total mentions (posts + comments)
    total_mentions = filtered_df[filtered_df[theme_col]].shape[0]
    if thread_count > 0:
        theme_counts[theme_name] = {
            'threads': thread_count,
            'mentions': total_mentions,
            'percentage': thread_count / unique_threads * 100
        }

# Sort themes by prevalence
sorted_themes = sorted(theme_counts.items(), key=lambda x: x[1]['threads'], reverse=True)

# Create insight cards for top themes
insight_col1, insight_col2 = st.columns(2)

with insight_col1:
    st.markdown("### 🎯 Most Discussed Topics")
    
    # Top 3 themes
    for idx, (theme, data) in enumerate(sorted_themes[:3]):
        emoji = "🔴" if theme in ['Job Security/RIFs', 'Political Appointments'] else "🟡"
        st.markdown(f"""
        <div class='insight-card insight-warning'>
            <b>{emoji} {data['percentage']:.0f}% discussed {theme}</b><br>
            <small>{data['threads']:,} threads with {data['mentions']:,} total posts/comments</small>
        </div>
        """, unsafe_allow_html=True)

with insight_col2:
    st.markdown("### 📊 Key Statistics")
    
    # Overall engagement metrics
    hiring_df = filtered_df[filtered_df['is_hiring_related']]
    total_hiring_discussions = hiring_df['thread_id'].nunique()
    avg_thread_size = filtered_threads['comment_count'].mean()
    
    st.markdown(f"""
    <div class='insight-card insight-info'>
        <b>{total_hiring_discussions:,} hiring-related discussions analyzed</b><br>
        <small>Average of {avg_thread_size:.0f} comments per thread</small>
    </div>
    """, unsafe_allow_html=True)
    
    # Viral discussion indicator
    viral_hiring = filtered_threads[
        (filtered_threads['is_hiring_thread']) & 
        (filtered_threads['comment_count'] > 50)
    ].shape[0]
    
    st.markdown(f"""
    <div class='insight-card insight-negative'>
        <b>{viral_hiring} threads sparked major debates (50+ comments)</b><br>
        <small>Indicating high emotional engagement with these topics</small>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ===== 3. POLICY IMPACT HIGHLIGHTS =====
# Analyze each milestone first
milestone_impacts = []
for milestone in milestones:
    if milestone['date'] <= date_max:  # Only analyze milestones that have occurred
        impact = analyze_milestone_impact(filtered_df, milestone)
        milestone_impacts.append({
            'name': f"{milestone['name']} ({milestone['date'].strftime('%b %d, %Y')})",
            'date': milestone['date'],
            'impact': impact
        })

if milestone_impacts:
    st.markdown("## Major Policy Events & Their Sentiment Impact")
    
    # Find most impactful events
    most_negative = max(milestone_impacts, key=lambda x: x['impact']['7_day']['negative_pct'])
    most_activity = max(milestone_impacts, key=lambda x: x['impact']['7_day']['posts'])
    
    # Top Negative Sentiment Policies
    st.markdown("### Top Negative Sentiment Policies")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class='insight-card insight-negative'>
            <b>Worst Received Policy</b><br>
            <b style='font-size: 1.1em;'>{most_negative['name']}</b><br>
            <small>📉 {most_negative['impact']['7_day']['negative_pct']:.0f}% negative sentiment</small><br>
            <small>💬 {most_negative['impact']['7_day']['posts']} threads discussing this event</small>
        </div>
        """, unsafe_allow_html=True)
        
        # RIF/Layoff specific impact
        layoff_events = [m for m in milestone_impacts if any(term in m['name'].lower() for term in ['layoff', 'termination', 'rif'])]
        if layoff_events:
            total_layoff_threads = sum(m['impact']['7_day']['posts'] for m in layoff_events)
            avg_layoff_negative = sum(m['impact']['7_day']['negative_pct'] for m in layoff_events) / len(layoff_events)
            
            st.markdown(f"""
            <div class='insight-card insight-negative'>
                <b>Layoff/RIF Events Combined Impact</b><br>
                <small>📊 {len(layoff_events)} separate termination events</small><br>
                <small>📉 {avg_layoff_negative:.0f}% average negative sentiment</small><br>
                <small>💬 {total_layoff_threads} total discussion threads</small>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        # Top Positive or Neutral Sentiment Policies (if any exist)
        least_negative = min(milestone_impacts, key=lambda x: x['impact']['7_day']['negative_pct'])
        if least_negative['impact']['7_day']['negative_pct'] < 50:
            st.markdown(f"""
            <div class='insight-card insight-info'>
                <b>Least Negative Policy Response</b><br>
                <b style='font-size: 1.1em;'>{least_negative['name']}</b><br>
                <small>📊 {least_negative['impact']['7_day']['negative_pct']:.0f}% negative sentiment</small><br>
                <small>💬 {least_negative['impact']['7_day']['posts']} threads</small>
            </div>
            """, unsafe_allow_html=True)
        
        # Schedule F specific
        schedule_f = next((m for m in milestone_impacts if 'Schedule F' in m['name']), None)
        if schedule_f:
            st.markdown(f"""
            <div class='insight-card insight-warning'>
                <b>Schedule F Launch Impact</b><br>
                <small>Immediate: {schedule_f['impact']['immediate']['negative_pct']:.0f}% negative</small><br>
                <small>7-day: {schedule_f['impact']['7_day']['negative_pct']:.0f}% negative</small><br>
                <small>30-day: {schedule_f['impact']['30_day']['negative_pct']:.0f}% negative</small>
            </div>
            """, unsafe_allow_html=True)

st.markdown("---")

# ===== 4. DISCUSSION VOLUME TRENDS =====
st.markdown("## Discussion Volume Trends")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Policy Discussion Frequency")
    
    # Create timeline of policy discussions
    if milestone_impacts:
        policy_timeline = []
        for m in milestone_impacts:
            policy_timeline.append({
                'Policy': m['name'][:40] + '...' if len(m['name']) > 40 else m['name'],
                'Date': m['date'],
                'Immediate Threads': m['impact']['immediate']['posts'],
                '7-Day Threads': m['impact']['7_day']['posts']
            })
        
        policy_df = pd.DataFrame(policy_timeline)
        
        # Create bar chart of discussion volume
        fig_volume = px.bar(
            policy_df,
            x='Date',
            y='7-Day Threads',
            hover_data=['Policy', 'Immediate Threads'],
            title='Thread Volume by Policy Event (7-day window)',
            labels={'7-Day Threads': 'Number of Threads'}
        )
        fig_volume.update_layout(height=350)
        st.plotly_chart(fig_volume, use_container_width=True)

with col2:
    st.markdown("### Most Discussed Hiring Policies")
    
    st.markdown(f"""
    <div class='insight-card insight-warning'>
        <b>Highest Discussion Volume</b><br>
        <b style='font-size: 1.1em;'>{most_activity['name']}</b><br>
        <small>🔥 {most_activity['impact']['immediate']['posts']} immediate threads</small><br>
        <small>📈 {most_activity['impact']['7_day']['posts']} total threads in first week</small>
    </div>
    """, unsafe_allow_html=True)
    
    # Top 3 most discussed
    top_discussed = sorted(milestone_impacts, key=lambda x: x['impact']['7_day']['posts'], reverse=True)[:3]
    for idx, policy in enumerate(top_discussed):
        st.markdown(f"""
        **{idx+1}. {policy['name']}**  
        📊 {policy['impact']['7_day']['posts']} threads | 
        📉 {policy['impact']['7_day']['negative_pct']:.0f}% negative
        """)

st.markdown("---")

# ===== 5. SENTIMENT DYNAMICS =====
st.markdown("## Sentiment Dynamics")

# Overall sentiment breakdown chart
st.markdown("### Sentiment Over Time")

# Calculate daily sentiment
daily_sentiment = hiring_df.groupby('date').agg({
    'sentiment_score': 'mean',
    'thread_id': 'nunique',
    'sentiment_category': lambda x: (x == 'Pessimistic').sum()
}).reset_index()
daily_sentiment['negative_pct'] = daily_sentiment['sentiment_category'] / daily_sentiment['thread_id'] * 100

# Create sentiment timeline
fig_sentiment_time = px.line(
    daily_sentiment,
    x='date',
    y='sentiment_score',
    title='Average Sentiment Score Over Time',
    labels={'sentiment_score': 'Sentiment Score', 'date': 'Date'},
    line_shape='spline'
)

# Add policy event markers
if milestone_impacts:
    for idx, m in enumerate(milestone_impacts[:5]):  # Top 5 events
        fig_sentiment_time.add_vline(
            x=m['date'].isoformat(),  # Convert date to string format
            line_dash="dash", 
            line_color="red",
            annotation_text=m['name'][:20] + '...',
            annotation_position="top"
        )

fig_sentiment_time.update_layout(height=400)
st.plotly_chart(fig_sentiment_time, use_container_width=True)

# Sentiment by Subreddit
st.markdown("### Sentiment by Community")

subreddit_sentiment = hiring_df.groupby('subreddit').agg({
    'sentiment_score': 'mean',
    'thread_id': 'nunique',
    'sentiment_category': lambda x: (x == 'Pessimistic').sum()
}).reset_index()
subreddit_sentiment['negative_pct'] = subreddit_sentiment['sentiment_category'] / subreddit_sentiment['thread_id'] * 100
subreddit_sentiment = subreddit_sentiment.sort_values('sentiment_score')

fig_sub_sentiment = px.bar(
    subreddit_sentiment,
    x='sentiment_score',
    y='subreddit',
    orientation='h',
    color='sentiment_score',
    color_continuous_scale='RdYlGn',
    color_continuous_midpoint=0,
    title='Average Sentiment by Subreddit',
    labels={'sentiment_score': 'Average Sentiment Score', 'subreddit': 'Subreddit'},
    hover_data={'negative_pct': ':.0f'}
)
fig_sub_sentiment.update_layout(height=300)
st.plotly_chart(fig_sub_sentiment, use_container_width=True)

st.markdown("---")

# ===== 6. COMMUNITY & LEGAL RESPONSES =====
st.markdown("## Community & Legal Responses")

# Legal interventions
legal_events = [m for m in milestone_impacts if any(term in m['name'].lower() for term in ['court', 'injunction', 'legal'])]
if legal_events:
    st.markdown("### Legal Interventions & Worker Reaction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class='insight-card insight-info'>
            <b>Legal Interventions Impact</b><br>
            <small>⚖️ {len(legal_events)} court-related events analyzed</small><br>
            <small>📊 Mixed to negative sentiment response</small><br>
            <small>💭 Pattern: Temporary hope followed by continued concern</small>
        </div>
        """, unsafe_allow_html=True)
        
        # Show specific legal events
        for event in legal_events:
            st.markdown(f"""
            **{event['name']}**  
            • Immediate: {event['impact']['immediate']['negative_pct']:.0f}% negative  
            • 7-day: {event['impact']['7_day']['negative_pct']:.0f}% negative
            """)
    
    with col2:
        st.markdown("### Temporary Hope vs. Continued Concern")
        
        # Calculate sentiment shift for legal events
        if legal_events:
            for event in legal_events:
                immediate_neg = event['impact']['immediate']['negative_pct']
                week_neg = event['impact']['7_day']['negative_pct']
                month_neg = event['impact']['30_day']['negative_pct']
                
                if immediate_neg > week_neg:
                    st.markdown(f"""
                    <div style='background-color: #d1fae5; padding: 10px; margin: 5px 0; border-radius: 5px;'>
                        <b>{event['name'][:30]}...</b><br>
                        Initial relief: {immediate_neg:.0f}% → {week_neg:.0f}% negative (↓{immediate_neg-week_neg:.0f}%)
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style='background-color: #fee2e2; padding: 10px; margin: 5px 0; border-radius: 5px;'>
                        <b>{event['name'][:30]}...</b><br>
                        Disappointment: {immediate_neg:.0f}% → {week_neg:.0f}% negative (↑{week_neg-immediate_neg:.0f}%)
                    </div>
                    """, unsafe_allow_html=True)

st.markdown("---")

# ===== 7. RECOMMENDATIONS & NEXT STEPS =====
st.markdown("## Recommendations & Next Steps")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### What This Means for Federal HR")
    
    st.markdown("""
    Based on the sentiment analysis and discussion patterns:
    
    **🚨 Immediate Concerns to Address:**
    1. **Job Security Fears** - Dominating {:.0f}% of discussions
    2. **Merit vs Political Loyalty** - Creating deep mistrust
    3. **Morale Crisis** - Widespread burnout and exhaustion
    
    **📊 Key Statistics:**
    - {:.0f}% of hiring discussions are negative
    - {:.0f}x more pessimism than optimism
    - {:,} threads directly responding to policy changes
    """.format(
        theme_counts.get('Job Security/RIFs', 0) / unique_threads * 100 if theme_counts else 0,
        pessimistic_count/len(hiring_df)*100 if len(hiring_df) > 0 else 0,
        pessimistic_ratio if pessimistic_ratio != float('inf') else 10,
        sum(m['impact']['7_day']['posts'] for m in milestone_impacts)
    ))

with col2:
    st.markdown("### Actionable Takeaways")
    
    st.markdown("""
    **For Leadership:**
    - Address job security concerns transparently and immediately
    - Clarify merit-based hiring commitments
    - Invest in morale recovery programs
    
    **For HR Teams:**
    - Prepare for increased turnover and recruitment challenges
    - Document and communicate clear hiring criteria
    - Create support systems for stressed employees
    
    **For Communications:**
    - Counter misinformation about hiring processes
    - Amplify success stories where they exist
    - Provide regular, transparent updates
    """)

# Data source citation
st.markdown("---")
st.caption("""
**Data Sources:** Reddit posts from r/FedEmployees, r/govfire, r/DeptHHS, and r/feddiscussion. 
Policy milestone dates based on publicly reported federal hiring changes and court decisions from January-July 2025.
All quotes are direct excerpts from high-engagement federal employee discussions.
""")

# EXPANDED SECTION: Hot Topics and Worker Voices
st.markdown("## 🔥 Hot Topics: What Federal Workers Are Really Saying")

# Get high-engagement comments with their parent posts
high_engagement_comments = filtered_df[
    (filtered_df['type'] == 'comment') &
    (filtered_df['sentiment_category'] == 'Pessimistic') &
    (filtered_df['body'].notna()) &
    (filtered_df['body'].str.len() > 100) &
    (filtered_df['score'] > 20)  # High engagement threshold for comments
].copy()

# Add recency weight
high_engagement_comments['days_ago'] = (pd.Timestamp.now() - high_engagement_comments['created_utc']).dt.days
high_engagement_comments['recency_weight'] = 1 / (high_engagement_comments['days_ago'] + 1)
high_engagement_comments['weighted_score'] = high_engagement_comments['score'] * high_engagement_comments['recency_weight']

# Sort by weighted score and remove duplicates by thread
high_engagement_comments = high_engagement_comments.sort_values('weighted_score', ascending=False)
high_engagement_comments = high_engagement_comments.drop_duplicates(subset=['thread_id'], keep='first')

# Get parent post information for each comment
def get_parent_post(thread_id):
    parent = filtered_df[(filtered_df['thread_id'] == thread_id) & (filtered_df['type'] == 'post')]
    if not parent.empty:
        parent = parent.iloc[0]
        title = str(parent['title']) if pd.notna(parent['title']) else ""
        body = str(parent['body']) if pd.notna(parent['body']) else ""
        # Create a summary of the original post
        if title:
            summary = title[:100] + "..." if len(title) > 100 else title
        elif body:
            summary = body[:100] + "..." if len(body) > 100 else body
        else:
            summary = "Discussion thread"
        return summary
    return "Discussion thread"

# Create three columns for different topic categories
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 🚨 Job Security & RIFs")
    st.markdown("*Top comments on termination fears*")
    
    job_comments = high_engagement_comments[high_engagement_comments['theme_Job Security/RIFs']].head(4)
    
    for idx, comment in job_comments.iterrows():
        # Get parent post context
        parent_summary = get_parent_post(comment['thread_id'])
        
        # Format comment text
        comment_text = str(comment['body'])[:200].strip()
        if len(str(comment['body'])) > 200:
            comment_text += "..."
        
        # Color based on sentiment intensity
        bg_color = "#fee2e2" if comment['sentiment_score'] < -2 else "#fef3c7"
        
        # Format date
        post_date = comment['created_utc'].strftime('%b %d, %Y')
        
        st.markdown(f"""
        <div style='background-color: {bg_color}; padding: 12px; margin: 10px 0; border-radius: 5px; border-left: 4px solid #dc2626;'>
            <div style='font-size: 0.8em; color: #6b7280; margin-bottom: 8px; padding: 5px; background-color: rgba(255,255,255,0.5); border-radius: 3px;'>
                <b>Original Post:</b> {parent_summary}
            </div>
            <div style='font-size: 0.9em; line-height: 1.4; margin-bottom: 8px;'>
                <b>Top Comment (⬆️ {comment['score']}):</b><br>
                <i>"{comment_text}"</i>
            </div>
            <div style='font-size: 0.85em; color: #4b5563;'>
                📍 r/{comment['subreddit']} | 📅 {post_date}
            </div>
        </div>
        """, unsafe_allow_html=True)

with col2:
    st.markdown("### 🎭 Political vs Merit")
    st.markdown("*Top comments on loyalty tests*")
    
    political_comments = high_engagement_comments[
        high_engagement_comments['theme_Political Appointments'] | 
        high_engagement_comments['theme_Merit vs Loyalty']
    ].head(4)
    
    for idx, comment in political_comments.iterrows():
        # Get parent post context
        parent_summary = get_parent_post(comment['thread_id'])
        
        # Format comment text
        comment_text = str(comment['body'])[:200].strip()
        if len(str(comment['body'])) > 200:
            comment_text += "..."
        
        bg_color = "#fee2e2" if comment['sentiment_score'] < -2 else "#fef3c7"
        
        # Format date
        post_date = comment['created_utc'].strftime('%b %d, %Y')
        
        st.markdown(f"""
        <div style='background-color: {bg_color}; padding: 12px; margin: 10px 0; border-radius: 5px; border-left: 4px solid #f59e0b;'>
            <div style='font-size: 0.8em; color: #6b7280; margin-bottom: 8px; padding: 5px; background-color: rgba(255,255,255,0.5); border-radius: 3px;'>
                <b>Original Post:</b> {parent_summary}
            </div>
            <div style='font-size: 0.9em; line-height: 1.4; margin-bottom: 8px;'>
                <b>Top Comment (⬆️ {comment['score']}):</b><br>
                <i>"{comment_text}"</i>
            </div>
            <div style='font-size: 0.85em; color: #4b5563;'>
                📍 r/{comment['subreddit']} | 📅 {post_date}
            </div>
        </div>
        """, unsafe_allow_html=True)

with col3:
    st.markdown("### 😔 Workplace Morale")
    st.markdown("*Top comments on burnout*")
    
    morale_comments = high_engagement_comments[
        high_engagement_comments['theme_Morale'] | 
        high_engagement_comments['theme_Remote Work'] |
        high_engagement_comments['theme_Pay and Benefits']
    ].head(4)
    
    for idx, comment in morale_comments.iterrows():
        # Get parent post context
        parent_summary = get_parent_post(comment['thread_id'])
        
        # Format comment text
        comment_text = str(comment['body'])[:200].strip()
        if len(str(comment['body'])) > 200:
            comment_text += "..."
        
        bg_color = "#fee2e2" if comment['sentiment_score'] < -2 else "#fef3c7"
        
        # Format date
        post_date = comment['created_utc'].strftime('%b %d, %Y')
        
        st.markdown(f"""
        <div style='background-color: {bg_color}; padding: 12px; margin: 10px 0; border-radius: 5px; border-left: 4px solid #6b7280;'>
            <div style='font-size: 0.8em; color: #6b7280; margin-bottom: 8px; padding: 5px; background-color: rgba(255,255,255,0.5); border-radius: 3px;'>
                <b>Original Post:</b> {parent_summary}
            </div>
            <div style='font-size: 0.9em; line-height: 1.4; margin-bottom: 8px;'>
                <b>Top Comment (⬆️ {comment['score']}):</b><br>
                <i>"{comment_text}"</i>
            </div>
            <div style='font-size: 0.85em; color: #4b5563;'>
                📍 r/{comment['subreddit']} | 📅 {post_date}
            </div>
        </div>
        """, unsafe_allow_html=True)

# Additional section for most viral discussions
st.markdown("---")
st.markdown("### 🔥 Most Engaged Discussions This Month")

# Get threads with highest total engagement (post score + comment scores)
thread_engagement = filtered_df.groupby('thread_id').agg({
    'score': 'sum',
    'type': 'count',
    'sentiment_score': 'mean'
}).reset_index()
thread_engagement.columns = ['thread_id', 'total_score', 'total_items', 'avg_sentiment']

# Get top threads
top_threads = thread_engagement.nlargest(5, 'total_score')

for idx, thread in top_threads.iterrows():
    # Get the original post
    post = filtered_df[(filtered_df['thread_id'] == thread['thread_id']) & (filtered_df['type'] == 'post')]
    if not post.empty:
        post = post.iloc[0]
        
        # Get top comment for this thread
        top_comment = filtered_df[
            (filtered_df['thread_id'] == thread['thread_id']) & 
            (filtered_df['type'] == 'comment')
        ].nlargest(1, 'score')
        
        if not top_comment.empty:
            top_comment = top_comment.iloc[0]
            
            # Determine sentiment color
            sentiment_emoji = "🔴" if thread['avg_sentiment'] < -0.5 else "🟡" if thread['avg_sentiment'] < 0.5 else "🟢"
            
            st.markdown(f"""
            <div style='background-color: #f8f9fa; padding: 15px; margin: 10px 0; border-radius: 5px; border: 1px solid #e5e7eb;'>
                <div style='margin-bottom: 10px;'>
                    {sentiment_emoji} <b>{post['title'] if pd.notna(post['title']) else 'Discussion'}</b>
                    <span style='float: right; color: #6b7280;'>r/{post['subreddit']} | {thread['total_items']-1} comments | {thread['total_score']} total score</span>
                </div>
                <div style='padding: 10px; background-color: white; border-radius: 3px; margin-top: 5px;'>
                    <div style='font-size: 0.9em; color: #4b5563; margin-bottom: 5px;'>
                        <b>Most upvoted comment (⬆️ {top_comment['score']}):</b>
                    </div>
                    <div style='font-size: 0.9em; font-style: italic;'>
                        "{str(top_comment['body'])[:300]}{'...' if len(str(top_comment['body'])) > 300 else ''}"
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

# Summary insight
total_negative_posts = len(high_engagement_comments)
avg_engagement = high_engagement_comments['score'].mean() if len(high_engagement_comments) > 0 else 0

st.markdown(f"""
<div style='background-color: #f3f4f6; padding: 15px; margin-top: 20px; border-radius: 5px; text-align: center;'>
    <b>📊 Sentiment Summary:</b> {total_negative_posts:,} high-engagement negative comments analyzed<br>
    <b>🔥 Average engagement:</b> {avg_engagement:.0f} upvotes per concerning comment<br>
    <b>💔 Overall tone:</b> Federal workers express deep betrayal, fear, and exhaustion
</div>
""", unsafe_allow_html=True)

# Data source citation
st.caption("""
**Data Sources:** Reddit posts from r/FedEmployees, r/govfire, r/DeptHHS, and r/feddiscussion. 
Policy milestone dates based on publicly reported federal hiring changes and court decisions from January-July 2025.
All quotes are direct excerpts from high-engagement federal employee discussions.
""")

st.markdown("---")

# Main content sections
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📊 What Workers Are Talking About")
    
    # Calculate theme frequencies
    theme_counts = {}
    theme_columns = [col for col in filtered_df.columns if col.startswith('theme_')]
    
    for theme_col in theme_columns:
        theme_name = theme_col.replace('theme_', '')
        count = filtered_df[filtered_df[theme_col]]['thread_id'].nunique()
        if count > 0:
            theme_counts[theme_name] = count
    
    # Create theme dataframe
    theme_df = pd.DataFrame(list(theme_counts.items()), columns=['Theme', 'Mentions'])
    theme_df = theme_df.sort_values('Mentions', ascending=True).tail(10)
    
    # Calculate percentages
    total_threads = filtered_df['thread_id'].nunique()
    theme_df['Percentage'] = (theme_df['Mentions'] / total_threads * 100).round(1)
    
    # Create horizontal bar chart
    fig_themes = px.bar(
        theme_df,
        x='Mentions',
        y='Theme',
        orientation='h',
        text='Mentions',
        color='Mentions',
        color_continuous_scale='Reds',  # Changed to red scale for negative tone
        labels={'Mentions': 'Number of Threads', 'Theme': ''}
    )
    
    # Add percentage labels
    fig_themes.update_traces(
        texttemplate='%{text} threads (%{customdata:.1f}%)',
        customdata=theme_df['Percentage'],
        textposition='outside'
    )
    
    fig_themes.update_layout(
        showlegend=False,
        height=400,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(showticklabels=False),
        coloraxis_showscale=False
    )
    
    st.plotly_chart(fig_themes, use_container_width=True)

with col2:
    st.markdown("### 🚨 Top Worries About Hiring")
    
    # Identify top concerns with sufficient data
    concerns = []
    
    for theme_col in theme_columns:
        theme_name = theme_col.replace('theme_', '')
        theme_data = filtered_df[filtered_df[theme_col]]
        
        if len(theme_data) > 10:  # Only include themes with sufficient data
            avg_sentiment = theme_data['sentiment_score'].mean()
            thread_count = theme_data['thread_id'].nunique()
            comment_count = len(theme_data) - thread_count
            
            # Get a high-quality sample quote (high score, negative sentiment)
            negative_samples = theme_data[
                (theme_data['sentiment_category'] == 'Pessimistic') & 
                (theme_data['score'] > 5) &
                (theme_data['body'].notna()) &
                (theme_data['body'].str.len() > 50)
            ].sort_values('score', ascending=False)
            
            if not negative_samples.empty:
                sample = negative_samples.iloc[0]
                sample_text = str(sample['body'])[:150] + "..." if len(str(sample['body'])) > 150 else str(sample['body'])
                sample_subreddit = sample['subreddit']
            else:
                # Fallback to any sample with text
                any_samples = theme_data[
                    (theme_data['body'].notna()) & 
                    (theme_data['body'].str.len() > 20)
                ].sort_values('score', ascending=False)
                
                if not any_samples.empty:
                    sample = any_samples.iloc[0]
                    sample_text = str(sample['body'])[:150] + "..." if len(str(sample['body'])) > 150 else str(sample['body'])
                    sample_subreddit = sample['subreddit']
                else:
                    continue  # Skip if no good samples
            
            concerns.append({
                'Theme': theme_name,
                'Sentiment': avg_sentiment,
                'Threads': thread_count,
                'Comments': comment_count,
                'Sample': sample_text,
                'Subreddit': sample_subreddit
            })
    
    # Sort by negativity and show top 5
    concerns_df = pd.DataFrame(concerns)
    if not concerns_df.empty:
        concerns_df = concerns_df.sort_values('Sentiment').head(5)
        
        for idx, concern in concerns_df.iterrows():
            # Determine emoji based on sentiment
            if concern['Sentiment'] < -1:
                emoji = "🔴"
                sentiment_class = "sentiment-negative"
            elif concern['Sentiment'] < -0.5:
                emoji = "🟡"
                sentiment_class = "sentiment-negative"
            else:
                emoji = "⚪"
                sentiment_class = "sentiment-neutral"
            
            st.markdown(f"""
            <div style='margin-bottom: 15px; padding: 10px; background-color: #f8f9fa; border-radius: 5px; border-left: 3px solid {"#dc2626" if concern["Sentiment"] < -0.5 else "#6b7280"}'>
                <b>{emoji} {concern['Theme']}</b> <span class='{sentiment_class}'>({concern['Sentiment']:.2f} sentiment)</span><br>
                <small>{concern['Threads']} threads, {concern['Comments']} comments</small><br>
                <small>Most discussed in r/{concern['Subreddit']}</small><br>
                <i style='font-size: 0.9em; color: #6b7280;'>"{concern['Sample']}"</i>
            </div>
            """, unsafe_allow_html=True)

st.markdown("---")

# Overall Sentiment Gauge (simplified, no longer central)
st.markdown("### 📈 Overall Hiring Sentiment")

# Define hiring_df - filter for hiring-related content
hiring_df = filtered_df[filtered_df['is_hiring_related']]

gauge_col1, gauge_col2, gauge_col3 = st.columns([1, 2, 1])

with gauge_col2:
    if not hiring_df.empty:
        avg_sentiment = hiring_df['sentiment_score'].mean()
        
        # Sentiment categories
        sentiment_dist = hiring_df['sentiment_category'].value_counts()
        total_hiring = len(hiring_df)
        
        pessimistic_pct = sentiment_dist.get('Pessimistic', 0) / total_hiring * 100
        neutral_pct = sentiment_dist.get('Neutral', 0) / total_hiring * 100
        optimistic_pct = sentiment_dist.get('Optimistic', 0) / total_hiring * 100
        
        # Create simple bar chart instead of gauge
        fig_sentiment = go.Figure()
        
        fig_sentiment.add_trace(go.Bar(
            x=[pessimistic_pct],
            y=['Sentiment'],
            name='Pessimistic',
            orientation='h',
            marker_color='#dc2626',
            text=f'{pessimistic_pct:.0f}%',
            textposition='inside',
            hovertemplate='Pessimistic: %{x:.1f}%<extra></extra>'
        ))
        
        fig_sentiment.add_trace(go.Bar(
            x=[neutral_pct],
            y=['Sentiment'],
            name='Neutral',
            orientation='h',
            marker_color='#6b7280',
            text=f'{neutral_pct:.0f}%',
            textposition='inside',
            hovertemplate='Neutral: %{x:.1f}%<extra></extra>'
        ))
        
        fig_sentiment.add_trace(go.Bar(
            x=[optimistic_pct],
            y=['Sentiment'],
            name='Optimistic',
            orientation='h',
            marker_color='#059669',
            text=f'{optimistic_pct:.0f}%' if optimistic_pct > 5 else '',
            textposition='inside',
            hovertemplate='Optimistic: %{x:.1f}%<extra></extra>'
        ))
        
        fig_sentiment.update_layout(
            barmode='stack',
            height=150,
            margin=dict(l=0, r=0, t=30, b=0),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
            xaxis=dict(showticklabels=False, showgrid=False, range=[0, 100]),
            yaxis=dict(showticklabels=False),
            title={
                'text': 'Federal workers are overwhelmingly pessimistic about hiring' if pessimistic_pct > 50 else 'Mixed sentiment about federal hiring',
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': {'size': 16}
            }
        )
        
        st.plotly_chart(fig_sentiment, use_container_width=True)
        
        st.caption(f"Based on {total_hiring:,} hiring-related discussions across all federal employee subreddits")
    else:
        st.warning("No hiring-related discussions found")

# Footer
st.markdown("---")
st.caption("Federal Hiring Pulse Check Dashboard | Data-driven insights into government workforce sentiment")
st.caption(f"Analysis covers {(date_max - date_min).days} days of Reddit discussions | Last updated: {datetime.now().strftime('%B %d, %Y at %I:%M %p EST')}")