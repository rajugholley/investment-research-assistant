# Investment Research Assistant

An AI-powered investment research tool that provides automated stock analysis and report generation.

## Features
- Real-time stock data fetching
- Technical analysis with interactive charts
- AI-powered investment insights
- Professional PDF report generation
- Moving averages and volatility analysis

## Technical Stack
- Streamlit for web interface
- Alpha Vantage API for market data
- OpenAI for analysis
- FPDF for PDF generation
- Plotly for interactive charts

## Setup Instructions
1. Clone the repository
2. Install requirements: `pip install -r requirements.txt`
3. Set up environment variables in .env:
   - ALPHA_VANTAGE_API_KEY
   - OPENAI_API_KEY
4. Run: `streamlit run investment_app.py`