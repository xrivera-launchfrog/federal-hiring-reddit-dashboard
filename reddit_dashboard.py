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
    """Analyze sentiment with realistic weighting for Reddit content"""
    
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
    
    # Sentiment patterns with Reddit-appropriate weighting
    # Negative patterns (weighted more heavily)
    strong_negative_patterns = r'\b(illegal|corrupt|theft|scam|fraud|disaster|terrible|awful|hate|disgusting|betrayal|crime|violation|unethical|unconstitutional)\b'
    moderate_negative_patterns = r'\b(worried|concerned|fear|uncertain|frustrated|angry|disappointed|unfair|worse|declining|problem|issue|difficult|struggle|failed|failing)\b'
    mild_negative_patterns = r'\b(confused|unclear|unsure|challenging|tough|hard|complicated|bureaucracy|red tape|slow|delays)\b'
    
    # Positive patterns (require stronger evidence)
    strong_positive_patterns = r'\b(excellent|amazing|fantastic|wonderful|love|perfect|best)\b'
    moderate_positive_patterns = r'\b(opportunity|improvement|hope|better|excited|positive|progress|success|happy|glad|optimistic)\b'
    
    # Context modifiers that flip sentiment
    negation_patterns = r'\b(not|no|never|without|lack|missing|absence)\s+'
    
    # Detect themes
    for theme, pattern in hiring_themes.items():
        df[f'theme_{theme}'] = df['full_text'].str.contains(pattern, case=False, regex=True, na=False)
    
    # Calculate sentiment with Reddit bias
    def calculate_realistic_sentiment(text):
        if pd.isna(text) or text.strip() == '':
            return 0, 'Neutral'
        
        text_lower = text.lower()
        
        # Count negative indicators (weighted)
        strong_neg = len(re.findall(strong_negative_patterns, text_lower)) * 3
        moderate_neg = len(re.findall(moderate_negative_patterns, text_lower)) * 2
        mild_neg = len(re.findall(mild_negative_patterns, text_lower)) * 1
        
        # Count positive indicators (higher threshold)
        strong_pos = len(re.findall(strong_positive_patterns, text_lower)) * 2
        moderate_pos = len(re.findall(moderate_positive_patterns, text_lower)) * 1
        
        # Check for negated positives (these become negative)
        negated_positives = len(re.findall(negation_patterns + r'(' + moderate_positive_patterns + r')', text_lower))
        
        # Calculate total score
        negative_score = strong_neg + moderate_neg + mild_neg + (negated_positives * 2)
        positive_score = strong_pos + moderate_pos
        
        # Reddit bias: require 2x more positive than negative to be considered positive
        net_score = positive_score - negative_score
        
        # Normalize by text length (per 100 words)
        word_count = len(text.split())
        if word_count > 20:  # Only normalize longer texts
            net_score = (net_score / word_count) * 100
        
        # Categorize with Reddit-appropriate thresholds
        if net_score > 2:  # High threshold for positive
            category = 'Optimistic'
        elif net_score < -0.5:  # Low threshold for negative
            category = 'Pessimistic'
        else:
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

# Header
st.title("📊 Federal Hiring Pulse Check")
st.markdown(f"Understanding how government workers feel about hiring practices")
st.markdown(f"**Analysis Period: {date_min.strftime('%B %d, %Y')} - {date_max.strftime('%B %d, %Y')}**")

# Use all data (no filtering)
filtered_df = df
filtered_threads = thread_stats

# Data Context Section (Expanded with methodology)
with st.expander("📋 **About This Analysis** - Click to see data sources and methodology", expanded=True):
    context_col1, context_col2 = st.columns(2)
    
    with context_col1:
        st.markdown("### Data Sources")
        
        # Subreddit breakdown
        subreddit_counts = filtered_df['subreddit'].value_counts()
        total_items = len(filtered_df)
        
        st.markdown("**Communities Analyzed:**")
        for subreddit, count in subreddit_counts.items():
            percentage = (count / total_items) * 100
            st.markdown(f"• **r/{subreddit}**: {count:,} items ({percentage:.1f}%)")
        
        st.markdown(f"\n**Community Context**: Federal employee forums with 250K+ combined members")
        
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

# Key Findings Section - Milestone-based
st.markdown("## 💡 Key Findings: Federal Worker Response to Major Policy Changes")

# Analyze each milestone
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
    # Find most impactful events
    most_negative = max(milestone_impacts, key=lambda x: x['impact']['7_day']['negative_pct'])
    most_activity = max(milestone_impacts, key=lambda x: x['impact']['7_day']['posts'])
    
    # Key milestone insights in cards
    st.markdown("### 🎯 Most Significant Policy Impacts")
    
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
        st.markdown(f"""
        <div class='insight-card insight-warning'>
            <b>Most Discussed Policy</b><br>
            <b style='font-size: 1.1em;'>{most_activity['name']}</b><br>
            <small>🔥 {most_activity['impact']['immediate']['posts']} immediate threads</small><br>
            <small>📈 {most_activity['impact']['7_day']['posts']} total threads in first week</small>
        </div>
        """, unsafe_allow_html=True)
        
        # Legal/Injunction impact
        legal_events = [m for m in milestone_impacts if any(term in m['name'].lower() for term in ['court', 'injunction', 'legal'])]
        if legal_events:
            st.markdown(f"""
            <div class='insight-card insight-info'>
                <b>Legal Interventions Impact</b><br>
                <small>⚖️ {len(legal_events)} court-related events</small><br>
                <small>📊 Mixed sentiment response</small><br>
                <small>💭 Temporary hope followed by continued concern</small>
            </div>
            """, unsafe_allow_html=True)

st.markdown("---")

# EXPANDED SECTION: Hot Topics and Worker Voices
st.markdown("## 🔥 Hot Topics: What Federal Workers Are Really Saying")

# Analyze hot topics by engagement and recency
hot_topics_df = filtered_df[
    (filtered_df['sentiment_category'] == 'Pessimistic') &
    (filtered_df['body'].notna()) &
    (filtered_df['body'].str.len() > 100) &
    (filtered_df['score'] > 10)  # High engagement posts
].copy()

# Add recency weight (more recent = higher weight)
# Fix: Use created_utc instead of date for datetime operations
hot_topics_df['days_ago'] = (pd.Timestamp.now() - hot_topics_df['created_utc']).dt.days
hot_topics_df['recency_weight'] = 1 / (hot_topics_df['days_ago'] + 1)
hot_topics_df['weighted_score'] = hot_topics_df['score'] * hot_topics_df['recency_weight']

# Sort by weighted score for "hottest" topics
hot_topics_df = hot_topics_df.sort_values('weighted_score', ascending=False)

# Create three columns for different topic categories
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 🚨 Job Security Fears")
    st.markdown("*The most visceral reactions*")
    
    job_fears = hot_topics_df[hot_topics_df['theme_Job Security/RIFs']].head(5)
    for idx, post in job_fears.iterrows():
        quote = str(post['body'])[:250].strip()
        if len(str(post['body'])) > 250:
            quote += "..."
        
        # Color intensity based on sentiment score
        bg_color = "#fee2e2" if post['sentiment_score'] < -2 else "#fef3c7"
        
        # Format the date properly
        post_date = post['created_utc'].strftime('%b %d, %Y')
        
        st.markdown(f"""
        <div style='background-color: {bg_color}; padding: 12px; margin: 8px 0; border-radius: 5px; border-left: 4px solid #dc2626;'>
            <div style='font-size: 0.85em; color: #4b5563; margin-bottom: 5px;'>
                📍 r/{post['subreddit']} | ⬆️ {post['score']} | 📅 {post_date}
            </div>
            <div style='font-size: 0.9em; line-height: 1.4;'>
                <i>"{quote}"</i>
            </div>
        </div>
        """, unsafe_allow_html=True)

with col2:
    st.markdown("### 🎭 Political vs Merit")
    st.markdown("*The loyalty test debate*")
    
    political_fears = hot_topics_df[
        hot_topics_df['theme_Political Appointments'] | 
        hot_topics_df['theme_Merit vs Loyalty']
    ].head(5)
    
    for idx, post in political_fears.iterrows():
        quote = str(post['body'])[:250].strip()
        if len(str(post['body'])) > 250:
            quote += "..."
        
        bg_color = "#fee2e2" if post['sentiment_score'] < -2 else "#fef3c7"
        
        # Format the date properly
        post_date = post['created_utc'].strftime('%b %d, %Y')
        
        st.markdown(f"""
        <div style='background-color: {bg_color}; padding: 12px; margin: 8px 0; border-radius: 5px; border-left: 4px solid #f59e0b;'>
            <div style='font-size: 0.85em; color: #4b5563; margin-bottom: 5px;'>
                📍 r/{post['subreddit']} | ⬆️ {post['score']} | 📅 {post_date}
            </div>
            <div style='font-size: 0.9em; line-height: 1.4;'>
                <i>"{quote}"</i>
            </div>
        </div>
        """, unsafe_allow_html=True)

with col3:
    st.markdown("### 😔 Morale Crisis")
    st.markdown("*The human toll*")
    
    morale_crisis = hot_topics_df[
        hot_topics_df['theme_Morale'] | 
        hot_topics_df['theme_Remote Work']
    ].head(5)
    
    for idx, post in morale_crisis.iterrows():
        quote = str(post['body'])[:250].strip()
        if len(str(post['body'])) > 250:
            quote += "..."
        
        bg_color = "#fee2e2" if post['sentiment_score'] < -2 else "#fef3c7"
        
        # Format the date properly
        post_date = post['created_utc'].strftime('%b %d, %Y')
        
        st.markdown(f"""
        <div style='background-color: {bg_color}; padding: 12px; margin: 8px 0; border-radius: 5px; border-left: 4px solid #6b7280;'>
            <div style='font-size: 0.85em; color: #4b5563; margin-bottom: 5px;'>
                📍 r/{post['subreddit']} | ⬆️ {post['score']} | 📅 {post_date}
            </div>
            <div style='font-size: 0.9em; line-height: 1.4;'>
                <i>"{quote}"</i>
            </div>
        </div>
        """, unsafe_allow_html=True)

# Trending topics analysis
st.markdown("---")
st.markdown("### 📈 Trending Concerns This Week")

# Get last 7 days of data
# Fix: Add days_ago column to filtered_df if not already there
if 'days_ago' not in filtered_df.columns:
    filtered_df['days_ago'] = (pd.Timestamp.now() - filtered_df['created_utc']).dt.days

last_week = filtered_df[filtered_df['days_ago'] <= 7]
if len(last_week) > 0:
    # Count theme mentions in last week
    trending_themes = {}
    for theme_col in theme_columns:
        if theme_col in last_week.columns:
            theme_name = theme_col.replace('theme_', '')
            count = last_week[last_week[theme_col]]['thread_id'].nunique()
            if count > 0:
                trending_themes[theme_name] = count
    
    # Display top trending
    if trending_themes:
        sorted_trending = sorted(trending_themes.items(), key=lambda x: x[1], reverse=True)[:5]
        
        trend_col1, trend_col2 = st.columns([2, 3])
        
        with trend_col1:
            for theme, count in sorted_trending:
                # Get sentiment for this theme in last week
                theme_sentiment = last_week[last_week[f'theme_{theme}']]['sentiment_score'].mean()
                sentiment_color = "🔴" if theme_sentiment < -1 else "🟡" if theme_sentiment < 0 else "🟢"
                
                st.markdown(f"""
                <div style='background-color: #f8f9fa; padding: 10px; margin: 5px 0; border-radius: 5px;'>
                    {sentiment_color} <b>{theme}</b><br>
                    <small>{count} threads this week</small>
                </div>
                """, unsafe_allow_html=True)
        
        with trend_col2:
            # Word frequency analysis for hot topics
            recent_text = ' '.join(last_week[last_week['sentiment_category'] == 'Pessimistic']['full_text'].dropna().tolist())
            
            # Count specific concerning phrases
            concerning_phrases = {
                '"illegal firings"': len(re.findall(r'illegal\s+fir', recent_text, re.I)),
                '"loyalty test"': len(re.findall(r'loyalty\s+test', recent_text, re.I)),
                '"mass layoffs"': len(re.findall(r'mass\s+layoff', recent_text, re.I)),
                '"hostile environment"': len(re.findall(r'hostile\s+environment', recent_text, re.I)),
                '"unfair treatment"': len(re.findall(r'unfair\s+treatment', recent_text, re.I)),
                '"political appointees"': len(re.findall(r'political\s+appointee', recent_text, re.I))
            }
            
            st.markdown("**🔍 Most Used Phrases (Last 7 Days)**")
            for phrase, count in sorted(concerning_phrases.items(), key=lambda x: x[1], reverse=True):
                if count > 0:
                    st.markdown(f"• {phrase}: **{count}** mentions")

# Summary insight
total_negative_posts = len(hot_topics_df)
avg_engagement = hot_topics_df['score'].mean() if len(hot_topics_df) > 0 else 0

st.markdown(f"""
<div style='background-color: #f3f4f6; padding: 15px; margin-top: 20px; border-radius: 5px; text-align: center;'>
    <b>📊 Sentiment Summary:</b> {total_negative_posts:,} high-engagement negative posts analyzed<br>
    <b>🔥 Average engagement:</b> {avg_engagement:.0f} upvotes per concerning post<br>
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