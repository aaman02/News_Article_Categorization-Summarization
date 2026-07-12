# 📰 News Article Categorization and Summarization using LLMs

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![LangChain](https://img.shields.io/badge/LangChain-Framework-green)
![Azure OpenAI](https://img.shields.io/badge/Azure%20OpenAI-LLM-orange)

## 🚀 Project Overview

This project demonstrates an end-to-end **AI-powered News Article Analysis System** using Large Language Models (LLMs).


## 🚀 Features

- **🔍 LLM-Based Classification**: Classify news articles into 5 categories using LLM
- **📝 AI Summarization**: Generate 60-word concise summaries using LLM
- **📰 Headline Generation**: Create catchy headline-style summaries
- **📊 Real-time Processing**: Get instant results for your articles

The system uses **LangChain + Azure OpenAI models** to perform intelligent news understanding and generation tasks.

---

## 🌐 Live Demo

🚀 Streamlit Application:

# 🎯 Problem Statement

With the increasing volume of digital news content, manually categorizing and summarizing articles is time-consuming.

The objective of this project is to build an AI system that can:

- Understand news article context
- Identify the primary topic
- Categorize articles accurately
- Generate meaningful summaries
- Produce human-like headlines

## 📋 Prerequisites

- Python 3.8+
- pip package manager

## 🛠️ Installation

1. **Navigate to the Streamlit folder:**
   ```bash
   cd Streamlit
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **(Optional) Set up LLM API key for live mode:**
   ```bash
   # Set environment variables
   set LLM_API_KEY=your_api_key_here
   set LLM_API_BASE_URL=https://your-api-endpoint.com
   ```

## ▶️ Running the Application

### Option 1: Direct Python Execution

```bash
streamlit run streamlit_app.py
```

The application will open in your default web browser at `http://localhost:8501`

### Option 2: Using Docker (Recommended for Production)

**Build and run with Docker Compose:**
```bash
docker-compose up --build
```

**Build and run with Docker directly:**
```bash
# Build the Docker image
docker build -t meridian-news-app .

# Run the container
docker run -p 8501:8501 meridian-news-app
```

**With environment variables (for live LLM API):**
```bash
docker run -p 8501:8501 \
  -e LLM_API_KEY=your_api_key \
  -e LLM_API_BASE_URL=https://your-api-endpoint.com \
  meridian-news-app
```

Access the application at `http://localhost:8501`

## 📁 Files

| File | Description |
|------|-------------|
| `streamlit_app.py` | Main Streamlit application with UI |
| `model.py` | LLM processor for classification and summarization |
| `utils.py` | Utility classes (fallback ML models) |
| `requirements.txt` | Python dependencies |
| `README.md` | This file |

## 🎯 How to Use

### Input Your Article

1. **Enter Headline**: Type or paste the article headline in the "Article Headline" field
2. **Enter Article**: Paste the full article content (supports 1000+ words) in the "Full Article" text area
3. **Click Process**: Click the "🚀 Process Article" button

### View Results

The application will display three outputs:

1. **🏷️ Category Classification**
   - Predicted category (Sports, Business, Technology, Education, or Entertainment)
   - Confidence score
   - Method used (LLM Zero-Shot Classification)

2. **📄 Concise Summary (60 words)**
   - 60-word summary capturing key points
   - Word count
   - Method (LLM Abstractive Summarization)

3. **📰 Headline-Style Summary**
   - Catchy headline version
   - Original headline for comparison

### Using Sample Articles

1. In the sidebar, select a sample category (Sports, Technology, Business, Education, Entertainment)
2. Click "Load Sample"
3. Click "Process Article" to see the results

## 🤖 Processing Modes

### Mock Mode (Default)
- ✅ Works without API key
- ✅ Perfect for demonstration
- ✅ Uses keyword-based classification and extractive summarization
- ✅ Simulates LLM behavior

### Live API Mode
- ⚠️ Requires LLM API key
- 🚀 Real LLM-powered processing
- 🚀 More accurate and contextual results
- 🚀 True abstractive summarization

## 🔧 Environment Variables

For live API mode, set these environment variables:

```bash
# Windows
set LLM_API_KEY=your_api_key_here
set LLM_API_BASE_URL=https://your-api-endpoint.com

# Linux/Mac
export LLM_API_KEY=your_api_key_here
export LLM_API_BASE_URL=https://your-api-endpoint.com
```

## 📊 Categories

The application classifies articles into:

| Category | Icon | Keywords |
|----------|------|----------|
| ⚽ Sports | 🏆 | game, team, player, score, championship |
| 💼 Business | 📈 | market, company, stock, economy, finance |
| 💻 Technology | 💡 | technology, AI, software, digital, computer |
| 🎓 Education | 📚 | education, school, student, learning, university |
| 🎭 Entertainment | 🎬 | movie, film, celebrity, music, entertainment |

## 🏗️ Architecture

```
User Input (Headline + Article)
        ↓
News Article Text
        ↓
Text Cleaning Pipeline
        ↓
LangChain Processing
        ↓
┌───────────────────────────────┐
│   LLMNewsProcessor (model.py) │
│  ┌─────────────────────────┐  │
│  │  Classification Prompt  │  │
│  │  Summarization Prompt   │  │
│  │  Headline Prompt        │  │
│  └─────────────────────────┘  │
│            ↓                  │
│  ┌─────────────────────────┐  │
│  │   LLM API (or Mock)     │  │
│  └─────────────────────────┘  │
└───────────────────────────────┘
        ↓
┌───────────────────────────────┐
│   Results Display             │
│  - Category Classification    │
│  - 60-Word Summary            │
│  - Headline Summary           │
└───────────────────────────────┘
```

## 📝 Example Input/Output

### Input
```
Headline: "Championship Team Wins Thrilling Final Match"

Article: The championship game ended in a thrilling victory as the 
home team scored a last-minute goal to secure their first title in 
over a decade. The crowd erupted in celebration as players rushed 
the field in jubilation... [continues for 200+ words]
```

### Output
```
🏷️ Category: Sports (85% confidence)

📄 Summary (60 words):
The championship game ended in a thrilling victory as the home team 
scored a last-minute goal. The crowd erupted in celebration as players 
rushed the field. The star player was named MVP after scoring two 
crucial goals. This marks the team's first championship win in over 
a decade, bringing joy to millions of fans worldwide.

📰 Headline:
Championship Team Secures Historic Victory with Last-Minute Goal
```

## 🔧 Troubleshooting

### Application won't start
```bash
# Reinstall dependencies
pip install -r requirements.txt --upgrade
```

### Mock mode not working
- Ensure `model.py` is in the same directory as `streamlit_app.py`
- Check that all imports are correct

### Live API not connecting
- Verify API key is set correctly
- Check API endpoint URL
- Ensure network connectivity

## 📈 Performance

| Metric | Mock Mode | Live API Mode |
|--------|-----------|---------------|
| Classification Accuracy | ~80% | ~90%+ |
| Summary Quality | Good | Excellent |
| Response Time | <1 second | 2-5 seconds |
| API Cost | Free | Depends on provider |

## 🙏 Acknowledgments

- Streamlit for the web framework
- NLTK for text processing utilities
- OpenAI/LLM providers for API access

## 📝 License

This project is part of a capstone project for educational purposes.

## 👥 Authors

Meridian Media Group Capstone Project (By Aman Bhargava)
