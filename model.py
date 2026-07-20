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
        return f"""You are an expert news article classifier with specialized knowledge in categorizing 
news articles across five domains: Sports, Business, Technology, Education, and Entertainment.

Classify the article into EXACTLY ONE category based on the content that you feel 
most confident about.

Available categories:
{', '.join(self.categories)}

=== CLASSIFICATION GUIDELINES ===

TECHNOLOGY:
Classify as 'technology' when the primary focus is:
- Technology/tech products (smartphones,platforms, computers, software, apps, gadgets,websites)
- AI/ML, algorithms, automation, digital services, artificial intelligence, machine learning, neural networks, LLMs
- Tech company product launches, features, or updates
- Cybersecurity, cloud computing, semiconductors, data centers, digital infrastructure
- Technical specifications, comparisons, or reviews
- Hardware: chips, semiconductors, smartphones, computers, devices, gadgets
- Tech companies' PRODUCT news: product launches, features, updates, platform changes
- Technical regulations: DMA, GDPR, app store policies, digital regulations

BUSINESS:
Classify as 'business' when the primary focus is:
- Financial metrics (stock prices, revenue, profits, earnings)
- Corporate actions (mergers, acquisitions, share transfers, IPOs)
- Market analysis, investments, funding rounds
- Executive appointments, business partnerships
- Economic policies, trade, industry trends
- Financial performance: revenue, profits, earnings reports, quarterly results
- Mergers, acquisitions, IPOs (unless primarily about tech product integration)
- Stock market: stock prices, trading, market movements, investor activities
- Economic policy: interest rates, inflation, trade policies
- General corporate strategy unrelated to specific tech products

SPORTS:
Classify as 'sports' when the primary focus is:
- Athletic competitions, games, matches, tournaments
- Teams, players, coaches, athletes
- Scores, championships, leagues, rankings
- Sports trades, transfers, team management
- Sports board/organization announcements

EDUCATION:
Classify as 'education' when the primary focus is:
- Schools, universities, colleges, academic institutions
- Entrance exams, eligibility, cut-offs, admissions
- Educational policies, curricula, learning methods
- Student achievements, academic research
- Teacher/faculty news, educational appointments

ENTERTAINMENT:
Classify as 'entertainment' when the primary focus is:
- Movies, films, TV shows, streaming content
- Music, concerts, albums, artists
- Celebrities, actors, directors, performers
- Box office, film releases, behind-the-scenes
- Gaming, books, literature, leisure content

=== DECISION APPROACH ===

Evaluate the article holistically and choose the category you feel most confident about.
Consider:
- What is the MAIN subject of the article?
- Who is the PRIMARY audience?
- What is the CENTRAL news being reported?

=== RULES IF CONFUSED ===
If confused between business and technology then follow below decision framework: (Only if confused between these two classes)
Ask yourself: "What is this article MAINLY about?"
- If "a technology product/platform/service" → technology
- If "money/financial performance/stock market" → business
- Even if the article mentions stock prices, earnings, CEOs, or lawsuits, classify as 'technology' if the MAIN topic is about a tech product, service, or platform.

=== EXAMPLES ===

Example 1 (TECHNOLOGY):
Headline: "OnePlus 12 vs Samsung Galaxy S24+: Which flagship phone wins?"
Content Key indicators: Smartphone comparison, hardware specs, processor, camera, display, battery, AI features
Category: Technology

Example 2 (SPORTS):
Headline: "India vs England: Harry Brook to fly home, will miss Test series"
Content: The England Cricket Board (ECB) on Sunday has confirmed that middle-order batsman Harry Brook will return to the UK and miss the India Test tour due to personal reasons.
â€œHarry Brook is set to return home with immediate effect for personal reasons from the England Menâ€™s Test tour of India. He will not be returning to India,â€ ECB said in a statement.
Key indicators: Cricket teams, player name, ECB statement, Test tour, sports board announcement
Category: Sports

Example 3 (ENTERTAINMENT):
Headline: "Mahesh Babu got migraine during Guntur Kaaram's shoot as he smoked 'real beedi'"
Content: Actor Mahesh Babuâ€™s latest film Guntur Kaaram has been doing well at the box office. The film features the actor in a rowdy avatar and is shown smoking a â€˜beediâ€™. Recently, the actor opened up about how smoking a â€˜beediâ€™ for the film affected his health. He also clarified that he doesnâ€™t â€˜encourageâ€™ smoking.
While speaking to Haarika and Hassine Creations, who backed the film, Mahesh Babu revealed that he suffered from migraines when he started shooting for Guntur Kaaram because of the â€˜beediâ€™. Then, the filmâ€™s director Trivikram Srinivas came to his aid and came up with an ayurvedic alternative.
You have exhausted your monthly limit of free stories.
Key indicators: Actor, film, box office, shooting, director, celebrity news
Category: Entertainment

Example 4 (EDUCATION):
Headline: "NEET MDS 2024 postponed to March 18; check cut-off date"
Content: The National Board of Examination in Medical Science (NBEMS) has postponed the National Eligibility-cum-Entrance Test for Master of Dental Surgery (NEET MDS 2024). As per the official notification, the NEET MDS 2024 will now be conducted on March 18.
NBEMS uploaded an official notification at the official website â€” natboard.edu.in.
The cut-off date for the purpose of eligibility to appear in the NEET MDS 2024 will be March 31, 2024. Earlier, the exam was scheduled to be tentatively conducted on February 9, 2024.
Read | NEET UG Toppersâ€™ Tips: â€˜Give mock tests to analyse mistakes and work on themâ€™
The question paper will consist of 240 multiple choice questions â€” Parts A and B consisting of 100 and 140 questions, respectively. Candidates will be awarded 4 marks for every correct answer and 1 mark will be deducted for every wrong answer.
Last year, the entrance test was scheduled to be conducted on March 1 and the results were declared around the end of March.
Meanwhile, NBEMS also postponed the NEET PG 2024 exam. According to the schedule released on January 9, the NEET PG 2024 exam will be conducted on July 7, 2024. The cut-off date for the purpose of eligibility to appear in the NEET PG 2024 will be August 15, 2024.
Key indicators: Entrance exam, National Board of Examination, eligibility, cut-off date, exam schedule
Category: Education

Example 5 (BUSINESS):
Headline: "Azim Premji gifts one crore Wipro shares to his sons"
Content: Wipro founder Azim Premji has transferred 1.02 crore equity shares of Wipro held by him to his two sons Rishad Premji and Tariq Premji as â€˜giftâ€™, according to an exchange filing.
The Wipro scrip is currently valued at Rs 472.9 per share, and at roughly this value, the transferred shares will amount to a whopping Rs 483 crore. Tech magnate Azim Premjiâ€™s son Rishad Premji currently helms Wipro as its Executive Chairman, and is a prominent face of the IT industry.
â€œI, Azim H Premji, wish to intimate you that 1,02,30,180 equity shares of Wipro Limited held by me, amounting to 0.20 per cent of the share capital of the company were transferred to Rishad Azim Premji and Tariq Azim Premji in the form of gift,â€ Wipro filing on Wednesday said.
ADVERTISEMENT
The transaction, however, would not alter the overall promoter and promoter group shareholding in the company and it will remain the same even after the proposed transaction.
In another filing by Wipro, Rishad Premji informed that 51,15,090 equity shares of Wipro Ltd has been received as a gift from Azim Premji.
A similar intimation was made for Tariq Premji, informing that he has also been gifted 51,15,090 equity shares of Wipro Ltd by Azim Premji.
Key indicators: Share transfer, equity shares, exchange filing, promoter shareholding, Executive Chairman
Category: Business

Example 6 (Business)
Headline: Zee, Sony yet to agree on merger conditions as deadline for extended negotiation nears
Content: The fate of the USD 10 billion merger between Zee Entertainment Enterprises and Culver Max Entertainment, formerly Sony Pictures Networks India, is hanging by a thread, with the two parties unable to finalise an agreement as the end of the one-month grace period looms.
The two parties are yet to come to an agreement over Zee Entertainment Enterprises Ltd (ZEEL) MD and CEO Punit Goenka leading the merged entity after Sony expressed concerns after market regulator Sebi barred him from holding managerial posts in Zee and any of the entities in a fund-diversion case.
Though the Securities and Exchange Board of India order was stayed by the Securities Appellate Tribunal, Sony is not comfortable with Goenka leading the merged entity due to the stringent corporate governance policy in Japan. The contentious issue is not just over Goenka leading the merged entity, but the completion of the deal also depends on how the Indian firm is able to meet the other closing conditions, said an industry source.
ADVERTISEMENT
The deal, which was signed between Zee Entertainment and Sony Pictures Networks India in 2021, has a stipulated period of two years in which the merger was to be completed before December 21, 2023, including regulatory and other approvals with a grace period of one month to complete the transaction.
Also Read | Japanâ€™s Sony says India unitâ€™s merger with Zee Entertainment likely delayed
By January 21, the one-month grace period for extended negotiations will end. Comments from Sony and ZEEL could not be obtained.
According to reports, the bone of contention is the driving seat of the merged entity. As per the agreed terms and conditions, Goenka was to lead the merger entity.
However, Culver Max Entertainment Pvt Ltd (CMEPL) is insisting on making way for its Sony Pictures Network head NP Singh.
ADVERTISEMENT
On December 17, the Subhash Chandra family promoted firm sought an extension of the December 21, 2023, deadline from Sony Group Corporation (SGC) firm Culver Max Entertainment and Bangla Entertainment Pvt Ltd (BEPL) under the Merger Cooperation Agreement dated December 22, 2021.
Earlier, Sony Pictures Networks India (SPNI) stated that it has not yet agreed to a deadline extension requested by ZEEL for their merger proposed USD 10-billion merger. However, a day after it agreed to discuss the matter. The proposed USD 10-billion merger of ZEEL, BEPL and CMEPL has already received regulatory approvals from the fair trade regulator CCI, bourses NSE and BSE, shareholders and creditors of the company.
In August this year, the Mumbai bench of the National Company Law Tribunal (NCLT) also gave a go-ahead to the merger of ZEEL and Culver Max Entertainment.
ADVERTISEMENT
This followed an interim order by Sebi barring Essel Group chairman Subhash Chandra and Zee Entertainment Enterprises Ltd MD and CEO Punit Goenka from holding the position of a director or key managerial personnel in any listed company. The market regulator took action after they were found diverting funds from the company.
Chandra and Goenka moved the Securities Appellate Tribunal (SAT) challenging the Sebi interim order. In October, SAT quashed the Sebi interim order.
Earlier in September 2021, then Sony Pictures Networks India and ZEEL entered into a non-binding term sheet to bring together their linear networks, digital assets, production operations and programme libraries. The combined entity will own over 70 TV channels, two video streaming services (ZEE5 and Sony LIV) and two film studios (Zee Studios and Sony Pictures Films India), making it the largest entertainment network in India.
Subsequently, the two parties signed a definitive agreement for their merger in December 2022. The majority of the board of directors of the combined entity would be nominated by the Sony Group and include the current SPNI Managing Director and CEO NP Singh.
However, questions over the future of the merger arose after Sebiâ€™s actions against Chandra and Goenka for siphoning off funds of ZEEL.
Category: Business
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
