import streamlit as st            # For creating the web interface
import pandas as pd              # For handling data in table format
from alpha_vantage.timeseries import TimeSeries  # To get stock market data
import plotly.graph_objects as go # For creating interactive charts
from datetime import datetime    # For handling dates
import os                       # For environment variables
from dotenv import load_dotenv  # For loading API keys safely
import openai                   # For calling GPT 3.5 LLM      
import numpy 
from fpdf import FPDF
import os
from datetime import datetime

class AnalysisReport(FPDF):
    def header(self):
        # Gholley Portfolio Advisory logo (using text as placeholder)
        self.set_font('Arial', 'B', 15)
        self.set_text_color(0, 51, 102)  # Dark blue color
        self.cell(0, 10, 'Gholley Portfolio Advisory', 0, 1, 'L')
        
        # Add company being analyzed
        if hasattr(self, 'company_symbol'):
            # Add some space
            self.ln(5)
            # Add company logo placeholder
            self.set_font('Arial', 'B', 12)
            self.cell(0, 10, f'{self.company_symbol} Analysis Report', 0, 1, 'C')
        
        # Line break
        self.ln(10)

    def footer(self):
        # Position at 1.5 cm from bottom
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        # Add page number
        self.cell(0, 10, f'Page {self.page_no()}/{{nb}}', 0, 0, 'C')

    def add_watermark(self, text):
        # Save current position
        x, y = self.get_x(), self.get_y()
        
        # Set watermark properties
        self.set_font('Arial', 'I', 50)
        self.set_text_color(200, 200, 200)  # Light gray
        
        # Rotate text
        self.rotate(45, self.w/2, self.h/2)
        self.text(30, 250, text)
        
        # Restore position and properties
        self.rotate(0)
        self.set_xy(x, y)
        self.set_text_color(0, 0, 0)  # Reset to black

def create_pdf_report(analysis, symbol):
    class PDF(FPDF):
        def header(self):
            # Gholley Portfolio Advisory branding
            self.set_font('Arial', 'B', 15)
            self.cell(0, 10, 'Gholley Portfolio Advisory', 0, 1, 'L')
            
            # Add some styling with colors
            self.set_draw_color(0, 51, 102)  # Navy blue
            self.set_line_width(0.5)
            self.line(20, 20, 190, 20)
            
            # Add company being analyzed
            self.ln(5)
            self.set_font('Arial', 'B', 12)
            self.cell(0, 10, f'Analysis Target: {symbol}', 0, 1, 'L')

        def footer(self):
            self.set_y(-15)
            self.set_font('Arial', 'I', 8)
            date_str = datetime.now().strftime("%Y-%m-%d")
            self.cell(0, 10, f'Page {self.page_no()} | Generated: {date_str}', 0, 0, 'C')

    # Create PDF object
    pdf = PDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_margins(20, 30, 20)

    # Title
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 20, f'{symbol} Investment Analysis', 0, 1, 'C')
    pdf.ln(5)

    def write_section(title, content):
        # Section title with styling
        pdf.set_font('Arial', 'B', 14)
        pdf.set_fill_color(240, 240, 240)  # Light gray background
        pdf.cell(0, 10, title, 0, 1, 'L', True)
        
        # Content
        pdf.set_font('Arial', '', 11)
        # Clean the content and handle encoding
        clean_content = content.encode('ascii', 'replace').decode('ascii').strip()
        pdf.ln(5)
        pdf.multi_cell(0, 7, clean_content)
        pdf.ln(5)

    # Process sections
    sections = {
        "Executive Summary": get_section_content(analysis, "EXECUTIVE SUMMARY", "INVESTMENT THESIS"),
        "Investment Thesis": get_section_content(analysis, "INVESTMENT THESIS", "RISK ASSESSMENT"),
        "Risk Assessment": get_section_content(analysis, "RISK ASSESSMENT", "RECOMMENDATION"),
        "Recommendation": get_section_content(analysis, "RECOMMENDATION")
    }

    for title, content in sections.items():
        write_section(title, content)

    # Save
    filename = f"{symbol}_analysis_report.pdf"
    pdf.output(filename)
    return filename

def get_section_content(text, start_marker, end_marker=None):
    if start_marker not in text:
        return ""
    
    parts = text.split(start_marker, 1)
    if len(parts) < 2:
        return ""
        
    content = parts[1]
    if end_marker:
        if end_marker in content:
            content = content.split(end_marker)[0]
    
    # Clean up the content by removing asterisks and extra whitespace
    content = content.replace('**', '').strip()
    # Clean up multiple newlines
    content = '\n'.join(line.strip() for line in content.split('\n') if line.strip())
    
    return content

# Page config - gives it a title and makes it use the full width of the screen.
st.set_page_config(
    page_title="Investment Research Assistant",
    layout="wide"
)
# Loads environment variables(Alpha_vantage API key in this case)
load_dotenv()
# Get API key
api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
openai.api_key = os.getenv('OPENAI_API_KEY')

def get_stock_data(symbol):
    """
    Fetch basic stock data using Alpha Vantage API
    """
    try:
        # Initialize TimeSeries with your API key
        ts = TimeSeries(key=api_key)
        
        # Get daily stock data
        data, meta_data = ts.get_daily(symbol=symbol)
        
        # Convert to DataFrame and prepare data
        df = pd.DataFrame.from_dict(data).T
        df.index = pd.to_datetime(df.index)
        
        # Rename columns for clarity
        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        # Convert strings to float
        for col in df.columns:
            df[col] = df[col].astype(float)
            
        return df
        
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None
    
def generate_stock_analysis(symbol, df, company_name):
    """
    Generate AI analysis focusing on investment thesis and risk assessment
    """
    # Prepare data points for analysis
    latest_price = df['Close'][0]
    price_change = ((df['Close'][0] - df['Close'][1]) / df['Close'][1]) * 100
    volatility = df['Close'].pct_change().std() * numpy.sqrt(252)  # Annualized volatility
    
    # Construct the prompt
    prompt = f"""
    As an investment analyst, provide a professional analysis of {company_name} ({symbol}).

    Key Data Points:
    - Current Price: ${latest_price:.2f}
    - Recent Price Change: {price_change:.2f}%
    - Volatility: {volatility:.2f}%
    
    Please provide your analysis in the following format:

    EXECUTIVE SUMMARY
    Provide a 2-3 sentence overview of your key findings and main recommendation.

    INVESTMENT THESIS
    1. Market Analysis
    - Current price trend and trading pattern
    - Market positioning and momentum
    
    2. Technical Analysis
    - Key price trends and patterns
    - Moving average implications
    - Volume analysis insights

    RISK ASSESSMENT
    1. Market Risks
    - Volatility analysis and patterns
    - Current market condition impacts
    
    2. Technical Risks
    - Support/resistance levels
    - Technical warning signs
    - Risk mitigation factors

    RECOMMENDATION
    Provide a clear, actionable recommendation including:
    - Time horizon
    - Entry/exit points
    - Key metrics to monitor
    
    Keep the language professional but clear, avoiding unnecessary jargon.

    """

    try:
        # Call OpenAI API using the new format
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model="gpt-4-0125-preview",
            messages=[
                {"role": "system", "content": "You are a very senior professional investment analyst writing clear, actionable analysis for portfolio managers."},
                {"role": "user", "content": prompt}
            ]
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating analysis: {str(e)}"
#Following code creates the User Interface Elements:

# Title
st.title("Investment Research Assistant")
st.subheader("Simple Stock Analysis Tool")

# Sidebar for user input
with st.sidebar:
    st.header("Settings")
    symbol = st.text_input("Enter Stock Symbol (e.g., AAPL, MSFT)", "AAPL")
    period = st.selectbox(
        "Select Time Period",
        ["1 month", "3 months", "6 months", "1 year", "5 year"]
    )

# Main content - # user clicks "Analyze" button and triggers the analysis 
# Main content
# Main content
if st.sidebar.button("Analyze"):
    st.info("Fetching data... Please wait.")
    try:
        # Get the data
        df = get_stock_data(symbol)
        
        if df is not None:
            st.success("Data fetched successfully!")
            
            # Display basic stock information
            st.header(f"Stock Data for {symbol}")
            
            # Format latest price and calculate daily change
            latest_price = df['Close'][0]
            previous_price = df['Close'][1]
            price_change = latest_price - previous_price
            percent_change = (price_change / previous_price) * 100
            
            # Display metrics with proper formatting
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Current Price", f"${latest_price:.2f}", 
                         f"{price_change:+.2f} ({percent_change:+.2f}%)")
            with col2:
                st.metric("Day's High", f"${df['High'][0]:.2f}")
            with col3:
                st.metric("Day's Low", f"${df['Low'][0]:.2f}")
            
            # Create price chart
            st.subheader("Price History")
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name='Price'
            ))
            
            # Update chart layout
            fig.update_layout(
                title=f'{symbol} Stock Price',
                yaxis_title='Price (USD)',
                xaxis_title='Date',
                template='none',  # Clean template
                height=500
            )
            
            # Display chart
            st.plotly_chart(fig, use_container_width=True)
            
            # Add Moving Averages to the price chart
            df['MA20'] = df['Close'].rolling(window=20).mean()
            df['MA50'] = df['Close'].rolling(window=50).mean()
            
            fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], name='20-day MA',
                                   line=dict(color='orange', width=1)))
            fig.add_trace(go.Scatter(x=df.index, y=df['MA50'], name='50-day MA',
                                   line=dict(color='blue', width=1)))
            
            # Display chart (keeping existing code)
            st.plotly_chart(fig, use_container_width=True)

            # Add Basic Statistics
            st.subheader("Basic Statistics")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Price Statistics**")
                stats_df = pd.DataFrame({
                    'Metric': ['Average Price', 'Highest Price', 'Lowest Price', 'Price Volatility'],
                    'Value': [
                        f"${df['Close'].mean():.2f}",
                        f"${df['High'].max():.2f}",
                        f"${df['Low'].min():.2f}",
                        f"{df['Close'].std():.2f}"
                    ]
                })
                st.dataframe(stats_df, hide_index=True)

            with col2:
                st.markdown("**Volume Analysis**")
                volume_df = pd.DataFrame({
                    'Metric': ['Average Volume', 'Highest Volume', 'Lowest Volume'],
                    'Value': [
                        f"{df['Volume'].mean():,.0f}",
                        f"{df['Volume'].max():,.0f}",
                        f"{df['Volume'].min():,.0f}"
                    ]
                })
                st.dataframe(volume_df, hide_index=True)

            # AI Analysis Section with better formatting
            st.subheader("Professional Investment Analysis")
            with st.spinner('Generating professional analysis...'):
                analysis = generate_stock_analysis(symbol, df, symbol)
                
                # Create tabs for different sections
                tab1, tab2, tab3, tab4 = st.tabs(["Executive Summary", "Investment Thesis", "Risk Assessment", "Recommendation"])
                
                # More robust section splitting
                def get_section(text, section_name):
                    try:
                        start = text.index(section_name)
                        next_section_names = ["EXECUTIVE SUMMARY", "INVESTMENT THESIS", "RISK ASSESSMENT", "RECOMMENDATION"]
                        # Remove the current section from potential next sections
                        next_section_names.remove(section_name)
                        
                        # Find the start of the next section
                        end = len(text)
                        for next_section in next_section_names:
                            try:
                                next_start = text.index(next_section)
                                if next_start > start and next_start < end:
                                    end = next_start
                            except ValueError:
                                continue
                        
                        # Extract and clean the section content
                        content = text[start:end].strip()
                        content = content.replace(section_name, "").strip()
                        return content
                    except ValueError:
                        return "Section not found"

                with tab1:
                    content = get_section(analysis, "EXECUTIVE SUMMARY")
                    st.markdown("### Executive Summary")
                    st.markdown(content)
                    
                with tab2:
                    content = get_section(analysis, "INVESTMENT THESIS")
                    st.markdown("### Investment Thesis")
                    st.markdown(content)
                    
                with tab3:
                    content = get_section(analysis, "RISK ASSESSMENT")
                    st.markdown("### Risk Assessment")
                    st.markdown(content)
                    
                with tab4:
                    content = get_section(analysis, "RECOMMENDATION")
                    st.markdown("### Recommendation")
                    st.markdown(content)
                
                # Add download buttons
            # Generate and offer PDF download
            # Add download buttons
            try:
                pdf_file = create_pdf_report(analysis, symbol)
                with open(pdf_file, "rb") as f:
                    st.download_button(
                        label="Download Analysis Report (PDF)",
                        data=f.read(),
                        file_name=f"{symbol}_analysis_report.pdf",
                        mime="application/pdf"
                    )
                os.remove(pdf_file)
            except Exception as e:
                st.error(f"PDF generation error: {str(e)}")
                
            except Exception as e:
                st.error(f"PDF generation error: {str(e)}")
                        # Show the data in a formatted table
                st.subheader("Recent Stock Data")
            # Format the dataframe
            formatted_df = df.copy()
            formatted_df = formatted_df.round(2)  # Round numbers to 2 decimal places
            formatted_df.index = formatted_df.index.strftime('%Y-%m-%d')  # Format dates
            st.dataframe(formatted_df.head(10))  # Show last 10 days
            
    except Exception as e:
        st.error(f"Error: {str(e)}")