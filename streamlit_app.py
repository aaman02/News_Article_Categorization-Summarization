"""
📰 News Article Classification and Summarization App
Built with Streamlit - LLM-Powered

This application provides:
1. News article classification into 5 categories using LLM
2. AI-powered summarization (60-word concise summaries and headline-style summaries)
3. Interactive demo with sample news articles
"""

import streamlit as st
import os
import sys

# Dynamically find the folder containing this streamlit_app.py file
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from model import LangChainNewsProcessor

# Page configuration
st.set_page_config(
    page_title="News Article Classification and Summarization App",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling and beautification
st.markdown("""
<style>
    /* Main header styling */
    .main-header {
        font-size: 2.8rem;
        font-weight: bold;
        background: linear-gradient(90deg, #1E3A8A 0%, #3B82F6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 0.5rem;
        padding: 1rem 0;
    }
    
    .sub-header {
        font-size: 1.3rem;
        color: #6B7280;
        text-align: center;
        margin-bottom: 2rem;
        font-style: italic;
    }
    
    /* Output section cards */
    .output-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 25px;
        margin: 15px 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border: none;
    }
    
    .output-section.category {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    }
    
    .output-section.summary {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    }
    
    .output-section.headline {
        background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
    }
    
    /* Category badges with dark text for visibility */
    .category-badge {
        display: inline-block;
        padding: 15px 35px;
        border-radius: 30px;
        font-weight: bold;
        font-size: 1.6rem;
        background-color: white;
        color: #1E3A8A;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    /* Button styling */
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        font-weight: bold;
        font-size: 1.1rem;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    /* Text styling inside cards */
    .output-section p {
        color: white;
        font-size: 1.15rem;
        line-height: 1.7;
        text-shadow: 1px 1px 3px rgba(0,0,0,0.2);
    }
    
    .output-section .headline-text {
        color: #1a1a2e;
        font-size: 1.5rem;
        font-weight: bold;
        text-align: center;
        text-shadow: none;
    }
    
    .output-section .meta-info {
        color: rgba(255,255,255,0.9);
        font-size: 0.95rem;
        margin-top: 12px;
    }
    
    /* Sidebar styling */
    .sidebar-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #1E3A8A;
        margin-bottom: 1rem;
    }
    
    /* Metric cards */
    .stMetric {
        background: linear-gradient(135deg, #e0e7ff 0%, #f0e6ff 100%);
        border-radius: 10px;
        padding: 15px;
        border: 2px solid #c7d2fe;
    }
    
    .stMetric label {
        color: #4c1d95;
        font-weight: bold;
    }
    
    .stMetric div[data-testid="stMetricValue"] {
        color: #1E3A8A;
        font-size: 1.8rem;
        font-weight: bold;
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.6rem;
        font-weight: bold;
        color: #1E3A8A;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #3B82F6;
    }
    
    /* Input section */
    .stTextInput>div>div>input, .stTextArea>div>div>textarea {
        border-radius: 10px;
        border: 2px solid #c7d2fe;
        font-size: 1rem;
    }
    
    .stTextInput>div>div>input:focus, .stTextArea>div>div>textarea:focus {
        border-color: #4f46e5;
        box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.1);
    }
    
    /* Word count caption */
    .word-count {
        background: linear-gradient(90deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 8px 15px;
        border-radius: 20px;
        display: inline-block;
        font-weight: bold;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# SESSION STATE
# -----------------------------

def initialize_session_state():
    """Initialize session state variables"""
    if 'llm_processor' not in st.session_state:
        st.session_state.llm_processor = LangChainNewsProcessor()
    if 'processing_complete' not in st.session_state:
        st.session_state.processing_complete = False
    if 'results' not in st.session_state:
        st.session_state.results = None


def main():
    """Main application"""
    
    # Initialize session state
    initialize_session_state()
    
    # Header with emoji
    st.markdown('<p class="main-header">📰 Meridian Media Group - News Article Categorization and Summarization</p>', 
                unsafe_allow_html=True)
    st.markdown('<p class="sub-header">✨ Powered by Large Language Models - AI-Powered News Analysis</p>', 
                unsafe_allow_html=True)
    
    st.divider()
    
    # Sidebar
    with st.sidebar:     
        # About
        st.markdown('<p class="sidebar-header">ℹ️ About</p>', unsafe_allow_html=True)
        st.markdown("""
        This application uses **LLM (Large Language Models)** to:
        - 📊 Classify news articles into categories
        - 📝 Generate 60-word concise summaries
        - 📰 Create headline-style summaries
        
        **Categories:**
        - ⚽ Sports
        - 💼 Business
        - 💻 Technology
        - 🎓 Education
        - 🎭 Entertainment
        
        **Input:**
        - Article Headline
        - Full Article (up to 1000+ words)
        - Press Process Article
        """)
        
        # st.divider()
        
        # Sample articles
        # st.markdown('<p class="sidebar-header">📋 Try Sample Articles</p>', unsafe_allow_html=True)
        
        # sample_articles = {
#             "Sports": {
#                 "headline": "Championship Team Wins Thrilling Final Match",
#                 "article": """The championship game ended in a thrilling victory as the home team scored a last-minute goal to secure their first title in over a decade. The crowd erupted in celebration as players rushed the field in jubilation. The star player, who scored two crucial goals during the match, was named the Most Valuable Player of the tournament. 

# The team's journey to the championship was filled with dramatic moments and impressive performances. Throughout the season, they demonstrated exceptional teamwork, strategic gameplay, and unwavering determination. The coaching staff praised the players for their dedication and hard work.

# Fans from around the world watched the match, with millions tuning in to witness the historic victory. The team's manager expressed gratitude to the supporters and dedicated the win to the fans who stood by them through thick and thin.

# This championship marks a new era for the team, bringing hope and excitement for future seasons. The victory parade is scheduled for next week, where the team will celebrate with their loyal fans in the city center."""
#             },
#             "Technology": {
#                 "headline": "Revolutionary AI System Transforms Industry",
#                 "article": """A groundbreaking artificial intelligence system has been unveiled by researchers, capable of solving complex mathematical problems faster than any previous technology. The new AI model uses advanced neural networks and has shown remarkable accuracy in various benchmarks across multiple domains.

# Tech companies are already exploring applications in healthcare and finance, where the system could revolutionize diagnostic processes and financial analysis. The development team spent three years perfecting the algorithm, which learns and adapts in ways that mimic human cognitive processes.

# Industry experts predict this breakthrough will accelerate innovation across multiple sectors. The AI system has already demonstrated capabilities in drug discovery, climate modeling, and autonomous vehicle navigation. Several major corporations have expressed interest in licensing the technology.

# The research team plans to publish their findings in a peer-reviewed journal and make the system available to academic researchers. This open approach aims to foster further innovation and collaboration in the AI community."""
#             },
#             "Business": {
#                 "headline": "Stock Markets Reach Record Highs",
#                 "article": """Stock markets reached record highs today as investor confidence surged following positive economic data releases. Major tech companies led the rally with significant gains, while traditional sectors also showed strong performance. Analysts predict continued growth in the coming quarters.

# Consumer spending remains robust despite inflation concerns, and unemployment rates stay at historically low levels. The Federal Reserve indicated it may adjust interest rates based on upcoming economic indicators. International markets also responded positively to the news.

# Financial experts recommend diversified investment strategies to capitalize on the growth while managing risk. The banking sector reported strong quarterly earnings, exceeding analyst expectations. Small business confidence has also improved, with many planning expansion in the near future.

# Economic forecasters remain cautiously optimistic about sustained growth, citing strong fundamentals and supportive monetary policies. However, they warn that global uncertainties could still impact market stability."""
#             },
#             "Education": {
#                 "headline": "New Study Reveals Benefits of Online Learning",
#                 "article": """A comprehensive new study reveals that online learning has become increasingly effective, with students showing improved engagement and retention rates compared to traditional classroom settings. Universities worldwide are adopting hybrid models that combine traditional teaching with digital tools.

# Educators emphasize the importance of personalized learning experiences enabled by technology. The study found that students appreciate the flexibility of online platforms, which allow them to learn at their own pace. Interactive features such as discussion forums and virtual labs have enhanced the learning experience.

# The research involved over 10,000 students across multiple institutions and disciplines. Results showed that well-designed online courses can achieve learning outcomes equal to or better than traditional methods. Student satisfaction rates have increased significantly with the implementation of new technologies.

# Universities are now investing heavily in digital infrastructure and faculty training to support hybrid education models. The trend is expected to continue as technology advances and student preferences evolve."""
#             },
#             "Entertainment": {
#                 "headline": "Blockbuster Movie Breaks Box Office Records",
#                 "article": """The highly anticipated movie premiere drew celebrities and fans alike to the red carpet event in what became one of the biggest entertainment spectacles of the year. Critics are praising the film's innovative storytelling and stunning visual effects.

# The director's latest work is expected to be a major contender in the upcoming awards season. The cast delivered outstanding performances, with particular acclaim for the lead actor's portrayal of the complex protagonist. The film's soundtrack has also gained popularity on music streaming platforms.

# Box office analysts predict the movie will break multiple records in its opening weekend. International audiences have responded enthusiastically, with strong ticket sales reported across major markets. The studio has already greenlit a sequel based on the positive reception.

# Fans have taken to social media to express their excitement, creating trending topics worldwide. The film's themes of hope and perseverance have resonated with audiences of all ages."""
#             }
#         }
        
#         selected_sample = st.selectbox("Select a sample article:", list(sample_articles.keys()))
        
#         if selected_sample:
#             if st.button("📥 Load Sample", use_container_width=True):
#                 st.session_state.sample_headline = sample_articles[selected_sample]["headline"]
#                 st.session_state.sample_article = sample_articles[selected_sample]["article"]
#                 st.success(f"✅ Loaded {selected_sample} sample!")
    
    # Main content area
    st.markdown('<p class="section-header">📝 Enter Article Details</p>', unsafe_allow_html=True)
    
    # Input section
    col1, col2 = st.columns([1, 2])
    
    with col1:
        headline_input = st.text_input(
            "📰 Article Headline",
            placeholder="Enter the article headline...",
            value=st.session_state.get('sample_headline', ''),
            key="headline_input"
        )
    
    # Article input
    article_input = st.text_area(
        "📄 Full Article Content",
        placeholder="Paste or type the full article content here (can be 1000+ words)...",
        height=250,
        value=st.session_state.get('sample_article', ''),
        key="article_input"
    )
    
    # Word count display
    word_count = len(article_input.split()) if article_input else 0
    st.markdown(f'<span class="word-count">📊 Article word count: {word_count} words</span>', 
                unsafe_allow_html=True)
    
    # Process button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        process_btn = st.button("🚀 Process Article", type="primary", use_container_width=True)
    with col2:
        clear_btn = st.button("🗑️ Clear All", use_container_width=True)
    with col3:
        if st.session_state.processing_complete:
            st.success("✅ Ready")
    
    # Clear functionality
    if clear_btn:
        st.session_state.processing_complete = False
        st.session_state.results = None
        st.session_state.sample_headline = ''
        st.session_state.sample_article = ''
        st.rerun()
    
    # Process article
    if process_btn:
        if not headline_input or not article_input:
            st.error("⚠️ Please enter both headline and article content!")
        elif len(article_input.split()) < 10:
            st.error("⚠️ Article is too short. Please enter a proper article (at least 10 words).")
        else:
            with st.spinner("🤖 LLM is processing your article..."):
                # Process using LLM
                results = st.session_state.llm_processor.process_article(
                    headline=headline_input,
                    content=article_input
                )
                
                st.session_state.results = results
                st.session_state.processing_complete = True
                st.success("✅ Article processed successfully!")
    
    # Display results
    if st.session_state.processing_complete and st.session_state.results:
        results = st.session_state.results
        
        st.markdown('<p class="section-header">📊 Analysis Results</p>', unsafe_allow_html=True)
        
        # Classification Result
        st.subheader("🏷️ Category Classification")
        # st.write("DEBUG CLASSIFICATION:", results['classification'])
        classification = results['classification']
        
        category = classification['category']
        
        # Category emoji mapping
        category_emoji = {
            'Sports': '⚽',
            'Business': '💼',
            'Technology': '💻',
            'Education': '🎓',
            'Entertainment': '🎭'
        }
        
        emoji = category_emoji.get(category, '📰')
        
        st.markdown(f"""
        <div class="output-section category">
            <div style="text-align: center; margin: 20px 0;">
                <span class="category-badge">
                    {emoji} {category}
                </span>
            </div>
            <p class="meta-info" style="text-align: center;">
                <strong>Confidence:</strong> {classification['confidence']:.0%} | 
                <strong>Method:</strong> {classification['method']}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Summary Result
        st.subheader("📄 Concise Summary (60 words)")
        summary = results['summary']
        
        st.markdown(f"""
        <div class="output-section summary">
            <p style="text-align: left;">{summary['summary']}</p>
            <p class="meta-info">
                <strong>📝 Word Count:</strong> {summary['word_count']} words | 
                <strong>⚙️ Method:</strong> {summary['method']}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Headline Result
        st.subheader("📰 Headline-Style Summary")
        headline_result = results['headline']
        
        st.markdown(f"""
        <div class="output-section headline">
            <p class="headline-text">"{headline_result['headline']}"</p>
            <p class="meta-info" style="text-align: center;">
                <strong>Original:</strong> {headline_result['original_headline']}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Article Statistics
        st.subheader("📊 Article Statistics")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📝 Total Words", results['article_length'])
        with col2:
            st.metric("🏷️ Category", category)
        with col3:
            st.metric("⚙️ Mode", results['processing_mode'].replace('_', ' ').title())
        with col4:
            st.metric("🎯 Confidence", f"{classification['confidence']:.0%}")


if __name__ == "__main__":
    main()
