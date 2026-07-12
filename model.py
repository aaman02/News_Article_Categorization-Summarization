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
                temperature=0,
                max_tokens=100
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
You are an expert news article classifier.

Classify the article into EXACTLY ONE category.

Available categories:
{', '.join(self.categories)}

Follow these rules strictly:

1. TECHNOLOGY has the highest priority when the article's PRIMARY SUBJECT is:
   - software, apps, platforms, websites
   - artificial intelligence, machine learning
   - hardware, devices, chips, semiconductors
   - smartphones, computers, gadgets
   - technology companies and their products/services
   - cybersecurity, cloud computing, digital infrastructure
   - technology regulations affecting tech products/platforms

   Even if the article mentions:
   - stock prices
   - company valuation
   - lawsuits
   - CEOs
   - earnings
   - investors
   
   classify as TECHNOLOGY if the main topic is about the technology product, service, or platform.

2. BUSINESS should only be selected when the PRIMARY SUBJECT is:
   - company earnings or financial performance
   - mergers and acquisitions
   - markets, stocks, investments
   - economic policy
   - business strategy unrelated to a specific technology product

3. Ask yourself:
   "What is this article mainly about?"

   If the answer is:
   "A technology product, platform, software, device, or digital service"
       -> Technology

   If the answer is:
   "Money, markets, finance, or general business operations"
       -> Business

Example 1:
Headline:
Spotify allows users to purchase subscriptions through its app in Europe

Reasoning:
The article discusses app store rules, digital platforms, and software services.

Category:
Technology


Example 2:
Headline:
Spotify reports record quarterly revenue and increased profits

Reasoning:
The article focuses on financial performance.

Category:
Business


Example 3:
Headline:
Apple launches a new AI-powered smartphone

Reasoning:
The article focuses on a technology product.

Category:
Technology

Example 4: 
Headline: Spotify to start in-app purchases on iPhone in Europe after DMA takes effect
Content: "Spotify users in Europe can start to buy audiobooks and subscription plans from within the music-streaming app from March as a result of the regionâ€™s new competition law for Big Tech, the Swedish company said on Wednesday.The move will help the company avoid Appleâ€™s 30% fee for purchases through its App Store, which has long been a source of contention between app developers and the tech giant.Spotify has for years been embroiled in a legal battle, alleging that it was forced to raise the price of its monthly subscriptions to cover costs tied to Appleâ€™s App Store rules. U.S.-listed shares of the Stockholm-based company rose around 2%. Also Read | What are smart rings and should you pick one over a fitness tracker? â€œFor years Apple had these rules where we couldnâ€™t tell you about offers, how much something costs, or even where or how to buy it,â€ Spotify said in a blogpost. â€œThe DMA (Digital Markets Act) means that weâ€™ll finally be able to share details about deals, promotions, and better-value payment options in the EU.â€ Under the DMA, which all Big Tech firms must comply with by March 7, companies are obligated to treat their own products and services like they do rivalsâ€™. ADVERTISEMENT Apple plans to challenge the European Unionâ€™s decision to put all of App Store into the blocâ€™s new digital antitrust list, Bloomberg News had reported in November.On Tuesday, Apple asked a London tribunal to throw out a mass lawsuit worth around $1 billion brought on behalf of more than 1,500 app developers over its App Store rules.Apple had also drawn criticism from Meta Platforms CEO Mark Zuckerberg who called App Store policies and fee structure as problematic and causing a conflict of interest.â€œWeâ€™ve always been interested in helping developers distribute their apps, and new options would add more competition in this space,â€ Meta said on Wednesday.ADVERTISEMENT â€œDevelopers deserve more ways to easily get their apps to the people that want them.â€"

Category:
Technology

Headline: {headline}

Article Content:
{content[:2000]}

Return ONLY one category name from:
{', '.join(self.categories)}
Category:"""
    
    def _build_summary_prompt(self, headline: str, content: str, max_words: int = 60) -> str:
        """Build prompt for summarization"""
        return f"""You are a professional news editor. Create a concise summary of the following news article.

Headline: {headline}

Article:
{content[:2000]}


Instructions:
- Write a clear, informative summary
- Use approximately {max_words} words or less
- Capture the key points and main message
- Write in a professional journalistic style

Summary:"""
    
    def _build_headline_prompt(self, headline: str, content: str) -> str:
        """Build prompt for headline-style summary"""
        return f"""You are a professional news editor. Create a catchy headline-style summary for the following article.

Original Headline: {headline}

Article:
{content[:2000]}

Instructions:
- Create a compelling headline that captures the essence
- Keep it under 20 words
- Make it engaging and informative
- Use newspaper headline style

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
        
        try:
            prompt = self._build_classification_prompt(headline, content)
            response = self.llm.invoke(prompt)
            category = response.content.strip().title()
            print(category)
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
        
        # Summarize
        summary = self.summarize(cleaned_headline, cleaned_content, 60)
        
        # Generate headline
        new_headline = self.generate_headline(cleaned_headline, cleaned_content)
        
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
