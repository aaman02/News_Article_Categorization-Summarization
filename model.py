"""
LangChain-based LLM Model for News Article Classification and Summarization
Uses Azure OpenAI services via LangChain libraries
"""

import os
import time
import warnings
from typing import Dict, List, Optional
from dotenv import load_dotenv

warnings.filterwarnings('ignore')
load_dotenv()
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

# Add these download safeguards so your application doesn't crash on startup
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# LangChain imports
try:
    from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
    from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("⚠️ LangChain not installed. Using mock mode.")
    
class AzureOpenAIConfig:
    """Configuration manager for Azure OpenAI services"""
    
    def __init__(self):
        self.api_key = os.getenv('AZURE_OPENAI_API_KEY')
        self.azure_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
        self.api_version = os.getenv('AZURE_OPENAI_API_VERSION')
        self.chat_deployment = os.getenv('AZURE_OPENAI_CHAT_DEPLOYMENT_NAME')
        
    def is_configured(self):
        """Check if properly configured with real API keys"""
        return bool(self.api_key and self.azure_endpoint and 'azure' in self.azure_endpoint.lower())


class LangChainNewsProcessor:
    """
    LangChain-powered processor for news article classification and summarization.
    
    Implements :
    Baseline: AzureChatOpenAI with zero-shot prompting
    """
    
    def __init__(self, config: Optional[AzureOpenAIConfig] = None, model_tier: str = 'baseline'):
        """
        Initialize the LangChain news processor.
        
        Args:
            config: AzureOpenAIConfig instance. If None, creates default config.
            model_tier: 'baseline', 'rag', or 'agent'
        """
        self.config = config or AzureOpenAIConfig()
        self.model_tier = model_tier
        self.use_mock = not (LANGCHAIN_AVAILABLE and self.config.is_configured())
        
        self.categories = ['Sports', 'Business', 'Technology', 'Education', 'Entertainment']
        self.stats = {
            'total_calls': 0,
            'total_tokens': 0,
            'total_time': 0,
            'success_count': 0,
            'error_count': 0
        }
        
        # Initialize LLM if not in mock mode
        self.llm = None
        self.embeddings = None
        self.agent = None
        
        if not self.use_mock:
            self._initialize_components()
    
    def _initialize_components(self):
        """Initialize LangChain components based on model tier"""
        try:
            # Initialize Azure Chat OpenAI
            self.llm = AzureChatOpenAI(
                azure_endpoint=self.config.azure_endpoint,
                api_key=self.config.api_key,
                api_version=self.config.api_version,
                deployment_name=self.config.chat_deployment,
                temperature=0.1,
                max_tokens=500,       # ⬆️ Increase for summaries/headlines
                top_p=0.95,           # Add for better quality
                frequency_penalty=0,  # Add to prevent repetition
                presence_penalty=0    # Add for balanced output
            )
            
            print(f"✅ LangChain components initialized (Tier: {self.model_tier})")
        except Exception as e:
            print(f"⚠️ Initialization failed: {e}. Using mock mode.")
            self.use_mock = True
    def clean_text(self, text: str) -> str:
        """
        Clean text before sending to LLM
        """
        # Lowercase
        text = text.lower()
    
        # Remove URLs
        text = re.sub(
            r'http\S+|www\S+',
            '',
            text
        )
    
        # Remove special characters
        text = re.sub(
            r'[^a-zA-Z\s]',
            '',
            text
        )
    
        # Remove extra spaces
        text = re.sub(
            r'\s+',
            ' ',
            text
        ).strip()
    
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        keep_words = {
            "not",
            "no",
            "never",
            "against"
        }
        stop_words = stop_words - keep_words
        words = [
            word 
            for word in text.split()
            if word not in stop_words
        ]
    
        return " ".join(words)  
      
    def _build_classification_prompt(self, headline: str, content: str) -> str:
        """Build prompt for classification"""        
        return f"""
You are an expert news article classifier with specialized knowledge in distinguishing 
technology articles from business articles.

Classify the article into EXACTLY ONE category.

Available categories:
{', '.join(self.categories)}

=== CRITICAL CLASSIFICATION RULES ===

## TECHNOLOGY PRIORITY RULE (HIGHEST PRIORITY)

Classify as TECHNOLOGY if ANY of these conditions are met:

1. **Product/Service Focus**: The article's main subject is a:
   - Software application, platform, website, or digital service
   - AI/ML model, algorithm, or automation system
   - Hardware device (smartphone, computer, chip, sensor, gadget)
   - Tech infrastructure (cloud, servers, networks, cybersecurity)

2. **Tech Company Product News**: Even if about a major tech company (Apple, Google, Microsoft, Meta, Amazon, Spotify, etc.), classify as TECHNOLOGY when discussing:
   - Product launches or updates
   - Feature releases or changes
   - Platform policies or rule changes
   - Technical capabilities or limitations
   - Software updates or OS changes
   - App Store policies and developer rules
   - Digital regulations affecting tech products (DMA, GDPR, etc.)

3. **Technology Keywords with High Weight**:
   - "AI", "artificial intelligence", "machine learning", "neural"
   - "app", "application", "software", "platform", "digital"
   - "device", "smartphone", "computer", "chip", "semiconductor"
   - "cybersecurity", "cloud", "algorithm", "data privacy"
   - "API", "SDK", "developer", "code", "programming"

## BUSINESS CLASSIFICATION (STRICT CRITERIA)

Classify as BUSINESS ONLY when the PRIMARY focus is:

1. **Financial Metrics**: Stock prices, earnings reports, revenue, profits, losses
2. **Corporate Actions**: Mergers, acquisitions, IPOs, layoffs, restructuring
3. **Market Analysis**: Market share, competition analysis, industry trends (non-tech specific)
4. **Investment/Finance**: Funding rounds, valuations, investor activities, stock market movements
5. **General Business Strategy**: Management changes, business partnerships (non-product related)

## DECISION FRAMEWORK

Ask these questions IN ORDER:

Q1: Is the article mainly about a technology product, feature, or technical capability?
    → If YES: Classify as TECHNOLOGY

Q2: Is the article mainly about financial performance, stock markets, or pure business operations?
    → If YES: Classify as BUSINESS

Q3: If the article mentions BOTH technology and business aspects:
    - If discussing HOW a technology works or WHAT it does → TECHNOLOGY
    - If discussing HOW MUCH money was made/lost → BUSINESS
    - If discussing regulatory rules affecting tech products → TECHNOLOGY
    - If discussing stock price reaction to news → BUSINESS

=== EXAMPLES ===

Example 1 (TECHNOLOGY - Platform Policy):
Headline: "Spotify allows users to purchase subscriptions through its app in Europe"
Key indicators: "app", "purchase through app", "Europe" (DMA regulation)
Category: Technology

Example 2 (BUSINESS - Financial Results):
Headline: "Spotify reports record quarterly revenue and increased profits"
Key indicators: "revenue", "profits", "quarterly"
Category: Business

Example 3 (TECHNOLOGY - Product Launch):
Headline: "Apple launches a new AI-powered smartphone with advanced camera"
Key indicators: "AI-powered", "smartphone", "camera" (product features)
Category: Technology

Example 4 (TECHNOLOGY - App Store Policy):
Headline: "Spotify to start in-app purchases on iPhone in Europe after DMA takes effect"
Key indicators: "in-app purchases", "iPhone", "DMA" (tech regulation)
Despite mentioning: legal battle, fees, shares rising 2%
Category: Technology (because main topic is app functionality and tech regulation)

Example 5 (BUSINESS - Stock Market):
Headline: "Tech stocks surge as market reacts to Fed interest rate decision"
Key indicators: "stocks surge", "market", "Fed", "interest rate"
Category: Business

Example 6 (TECHNOLOGY - Despite Business Elements):
Headline: "Google announces new AI model, shares rise 3%"
Key indicators: "AI model" is the primary announcement
Despite mentioning: "shares rise"
Category: Technology (product announcement is the main news, stock reaction is secondary)

=== YOUR TASK ===

Headline: {headline}

Article Content:
{content[:2000]}

=== OUTPUT FORMAT ===
- Return ONLY the category name in UPPERCASE
- Do not include any explanation or reasoning
- Do not include punctuation or additional text
- Valid outputs are ONLY: SPORTS, BUSINESS, TECHNOLOGY, EDUCATION, ENTERTAINMENT

Category:"""
  
    def _build_summary_prompt(self, headline: str, content: str, max_words: int = 60) -> str:
        """Build prompt for summarization"""
        return f"""You are a professional news editor. Create a concise summary of the following news article.

Headline: {headline}

Article:
{content[:2000]}

=== INSTRUCTIONS ===
1. Write a clear, informative summary capturing the key points and main message
2. Use approximately {max_words} words (acceptable range: 50-{max_words} words)
3. Maintain proper sentence structure with correct capitalization and punctuation
4. Keep all proper nouns (names, companies, places) intact
5. Write in professional journalistic style - objective and factual
6. Include the most important: who, what, when, where, why (if applicable)
7. Do not add opinions or information not in the original article

=== OUTPUT FORMAT ===
- Write in complete sentences with proper grammar
- Maintain original capitalization (do not convert to lowercase)
- Include necessary stopwords for readability
- Do not add any introductory phrases like "This article discusses..."

Summary:"""
    
    def _build_headline_prompt(self, headline: str, content: str) -> str:
        """Build prompt for headline-style summary"""
        return f"""You are a professional news editor. Create a catchy headline-style summary for the following article.

Original Headline: {headline}

Article content:
{content[:2000]}

=== INSTRUCTIONS ===
1. Create a compelling headline that captures the essence of the article
2. Keep it under 20 words (ideal: 10-15 words)
3. Use newspaper headline style:
   - Present tense for recent events
   - Omit unnecessary articles (a, an, the) when appropriate
   - Use strong, active verbs
   - Make it engaging and informative
4. Maintain proper capitalization (Title Case or Sentence case)
5. Preserve key proper nouns (names, companies, products)
6. Avoid clickbait - be accurate and factual

=== EXAMPLES ===
Good: "Spotify Launches In-App Purchases in Europe After DMA Regulation"
Good: "AI System Breakthrough Promises Faster Drug Discovery"
Bad: "You Won't Believe What This AI Can Do Now!" (clickbait)

=== OUTPUT FORMAT ===
- Return ONLY the headline text
- No quotation marks around the headline
- No additional explanation

New Headline:"""
    
    def _mock_classify(self, headline: str, content: str) -> str:
        """Mock classification for demo without API key"""
        text = (headline + " " + content).lower()
        
        scores = {
            'Sports': sum(text.count(w) for w in ['game', 'team', 'player', 'score', 'championship', 'win', 'match', 'tournament', 'athlete']),
            'Business': sum(text.count(w) for w in ['market', 'company', 'stock', 'business', 'economy', 'finance', 'investment', 'trade']),
            'Technology': sum(text.count(w) for w in ['technology', 'tech', 'software', 'digital', 'ai', 'computer', 'app', 'algorithm', 'data']),
            'Education': sum(text.count(w) for w in ['education', 'school', 'student', 'learning', 'university', 'college', 'teacher', 'academic']),
            'Entertainment': sum(text.count(w) for w in ['movie', 'film', 'celebrity', 'music', 'entertainment', 'actor', 'show', 'concert', 'streaming'])
        }
        
        best_category = max(scores, key=scores.get)
        return best_category if scores[best_category] > 0 else 'Technology'
    
    def _mock_summary(self, headline: str, content: str, max_words: int = 60) -> str:
        """Mock summary generation"""
        sentences = content.split('. ')
        if len(sentences) >= 3:
            summary = '. '.join(sentences[:3]) + '.'
            words = summary.split()
            if len(words) > max_words:
                summary = ' '.join(words[:max_words]) + '...'
            return summary
        return content[:500] + '...' if len(content) > 500 else content
    
    def classify(self, headline: str, content: str) -> Dict:
        """
        Classify a news article using LangChain.
        
        Args:
            headline: Article headline
            content: Full article text
            
        Returns:
            Dictionary with category and metadata
        """
        start_time = time.time()
        self.stats['total_calls'] += 1

        # Handle edge cases gracefully (don't raise errors)
        if not headline or not content:
            return {
                "category": "Unknown",
                "confidence": 0.0,
                "response_time": 0,
                "tokens_used": 0,
                "model_tier": self.model_tier,
                "using_mock": True,
                "method": "Error - Empty Input",
                "error": "Headline and content cannot be empty"
            }    
        # If content is too short, still try to classify
        if len(content) < 50:
            # Proceed with classification anyway
            # The LLM can handle short content
            pass    
        
        try:
            prompt = self._build_classification_prompt(headline, content)
            response = self.llm.invoke(prompt)
            category = response.content.strip().title()
            print(category)

            # Extract just the category if response contains extra text
            for cat in self.categories:
                if cat.lower() in category.lower():
                    category = cat
                    break          

            # Validate category
            if category not in self.categories:
                category = self._mock_classify(headline, content)
                print("not in category")
            tokens_used = len(prompt.split()) + len(response.content.split())
            self.stats['success_count'] += 1
        except Exception as e:
            print(f"Classification error: {e}")
            category = self._mock_classify(headline, content)
            tokens_used = 50
            self.stats['error_count'] += 1
        
        elapsed_time = time.time() - start_time
        self.stats['total_tokens'] += tokens_used
        self.stats['total_time'] += elapsed_time

        return {
            "category": category,
            "confidence": 0.85 if self.use_mock else 0.90,
            "response_time": elapsed_time,
            "tokens_used": tokens_used,
            "model_tier": self.model_tier,
            "using_mock": self.use_mock,
            "method": "Mock" if self.use_mock else "GPT"
        }
    def summarize(self, headline: str, content: str, max_words: int = 60) -> Dict:
        """
        Generate a concise summary using LangChain.
        
        Args:
            headline: Article headline
            content: Full article text
            max_words: Maximum words in summary
            
        Returns:
            Dictionary with summary and metadata
        """
        start_time = time.time()
        
        if self.use_mock:
            summary = self._mock_summary(headline, content, max_words)
            tokens_used = 100
        else:
            try:
                prompt = self._build_summary_prompt(headline, content, max_words)
                response = self.llm.invoke(prompt)
                summary = response.content.strip()
                
                # Ensure summary is within word limit
                words = summary.split()
                if len(words) > max_words:
                    summary = ' '.join(words[:max_words]) + '...'
                
                tokens_used = len(prompt.split()) + len(response.content.split())
            except Exception as e:
                print(f"Summarization error: {e}")
                summary = self._mock_summary(headline, content, max_words)
                tokens_used = 100
        
        elapsed_time = time.time() - start_time
        
        return {
            'summary': summary,
            'word_count': len(summary.split()),
            'max_words': max_words,
            'response_time': elapsed_time,
            'tokens_used': tokens_used,
            'using_mock': self.use_mock,
            "method": "Mock" if self.use_mock else "GPT"
        }
    
    def generate_headline(self, headline: str, content: str) -> Dict:
        """
        Generate a headline-style summary using LangChain.
        
        Args:
            headline: Original article headline
            content: Full article text
            
        Returns:
            Dictionary with new headline
        """
        start_time = time.time()
        
        if self.use_mock:
            words = headline.split()
            new_headline = ' '.join(words[:12]) + '...' if len(words) > 12 else headline
            tokens_used = 30
        else:
            try:
                prompt = self._build_headline_prompt(headline, content)
                response = self.llm.invoke(prompt)
                new_headline = response.content.strip()
                tokens_used = len(prompt.split()) + len(response.content.split())
            except Exception as e:
                print(f"Headline generation error: {e}")
                new_headline = headline
                tokens_used = 30
        
        elapsed_time = time.time() - start_time
        
        return {
            'headline': new_headline,
            'original_headline': headline,
            'response_time': elapsed_time,
            'tokens_used': tokens_used,
            'using_mock': self.use_mock
        }
    
    def process_article(self, headline: str, content: str) -> Dict:
        """
        Process an article completely - classify and summarize.
        
        Args:
            headline: Article headline
            content: Full article text            
        Returns:
            Dictionary with all results
        """

        # Clean article text before LLM processing
        cleaned_headline = self.clean_text(headline)
        cleaned_content = self.clean_text(content)
    
        # Classify
        classification = self.classify(cleaned_headline, cleaned_content)
        
        # Summarize use ORIGINAL text for natural output
        summary = self.summarize(headline, content, 60)
        
        # Generate headline use ORIGINAL text for natural output
        new_headline = self.generate_headline(headline, content)
        
        return {
            'classification': classification,
            'summary': summary,
            'headline': new_headline,
            'article_length': len(content.split()),
            'processing_mode': 'mock' if self.use_mock else 'live_api',
            'model_tier': self.model_tier,
            'stats': self.stats.copy()
        }
    
    def get_stats(self) -> Dict:
        """Get processing statistics"""
        if self.stats['total_calls'] == 0:
            return {}
        return {
            'total_calls': self.stats['total_calls'],
            'success_count': self.stats['success_count'],
            'error_count': self.stats['error_count'],
            'avg_response_time': self.stats['total_time'] / self.stats['total_calls'],
            'avg_tokens_per_call': self.stats['total_tokens'] / self.stats['total_calls'],
            'total_tokens': self.stats['total_tokens'],
            'success_rate': self.stats['success_count'] / self.stats['total_calls']
        }


# Create default instance
config = AzureOpenAIConfig()
llm_processor = LangChainNewsProcessor(config=config, model_tier='baseline')
