import streamlit as st
import requests
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import GooglePlacesAPIWrapper, SerpAPIWrapper, GoogleSerperAPIWrapper
from langchain_google_community import GooglePlacesTool
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import MessagesState, StateGraph, END, START
from langgraph.prebuilt import ToolNode, tools_condition
from typing import Dict, List, Any, Literal
import json
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- Classes from the notebook ---

class Config:
    def __init__(self):
        self.open_api_key = os.getenv('OPENAI_API_KEY')
        self.openweather_api_key = os.getenv('OPENWEATHER_API_KEY')
        self.exchange_rate_api_key = os.getenv('EXCHANGE_RATE_API_KEY')
        self.google_places_api_key = os.getenv('GOOGLE_PLACES_API_KEY')
        self.serpapi_key = os.getenv('SERPAPI_KEY')
        self.serper_api_key = os.getenv('SERPER_API_KEY')

class WeatherService:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "http://api.openweathermap.org/data/2.5"

    def get_current_weather(self, city: str) -> Dict:
        try:
            url = f"{self.base_url}/weather"
            params = {"q": city, "appid": self.api_key, "units": "metric"}
            response = requests.get(url, params=params)
            return response.json() if response.status_code == 200 else {}
        except:
            return {}

    def get_weather_forecast(self, city: str, days: int = 5) -> Dict:
        try:
            url = f"{self.base_url}/forecast"
            params = {"q": city, "appid": self.api_key, "units": "metric", "cnt": days * 8}
            response = requests.get(url, params=params)
            return response.json() if response.status_code == 200 else {}
        except:
            return {}

class CurrencyService:
    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.base_url = "https://api.exchangerate-api.com/v4/latest"

    def get_exchange_rate(self, from_currency: str, to_currency: str) -> float:
        try:
            url = f"{self.base_url}/{from_currency}"
            response = requests.get(url)
            data = response.json()
            if response.status_code == 200 and to_currency in data['rates']:
                return data['rates'][to_currency]
            return 1.0
        except:
            return 1.0

    def convert_currency(self, amount: float, from_currency: str, to_currency: str) -> float:
        rate = self.get_exchange_rate(from_currency, to_currency)
        return amount * rate

class TravelCalculator:
    @staticmethod
    def add(a: float, b: float) -> float:
        return a + b

    @staticmethod
    def multiply(a: float, b: float) -> float:
        return a * b

    @staticmethod
    def calculate_total_cost(*costs: float) -> float:
        return sum(costs)

    @staticmethod
    def calculate_daily_budget(total_cost: float, days: int) -> float:
        return total_cost / days if days > 0 else 0

class TravelPlanner:
    def __init__(self, config: Config):
        self.config = config
        self.weather_service = WeatherService(config.openweather_api_key)
        self.currency_service = CurrencyService(config.exchange_rate_api_key)
        self.calculator = TravelCalculator()
        self.search_tool = DuckDuckGoSearchRun()
        
        try:
            if config.google_places_api_key:
                places_wrapper = GooglePlacesAPIWrapper(google_places_api_key=config.google_places_api_key)
                self.places_tool = GooglePlacesTool(api_wrapper=places_wrapper)
            else:
                self.places_tool = None
        except Exception:
            self.places_tool = None
            
        try:
            if config.serpapi_key:
                self.serp_search = SerpAPIWrapper(serpapi_api_key=config.serpapi_key)
            else:
                self.serp_search = None
        except Exception:
            self.serp_search = None
            
        try:
            if config.serper_api_key:
                self.serper_search = GoogleSerperAPIWrapper(serper_api_key=config.serper_api_key)
            else:
                self.serper_search = None
        except Exception:
            self.serper_search = None
        
        self.llm = ChatOpenAI(
            base_url=os.getenv("BASE_URL"),
            model="openai/gpt-4.1",
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )
        
        self.tools = self._setup_tools()
        self.llm_with_tools = self.llm.bind_tools(self.tools)

    def _setup_tools(self) -> List:
        """Setup all tools for the travel agent"""
        
        @tool
        def search_images(query: str) -> List[str]:
            """Search for images related to a query."""
            if self.serper_search:
                try:
                    results = self.serper_search.results(query, type="images")
                    if 'images' in results:
                        return [img['imageUrl'] for img in results['images'][:5]]
                except Exception:
                    return []
            return []

        @tool
        def search_attractions(city: str) -> str:
            """Search for top attractions in a city using real-time data"""
            query = f"top attractions activities things to do in {city}"
            
            # Try Google Places first for real-time data
            if self.places_tool:
                try:
                    places_result = self.places_tool.run(f"tourist attractions in {city}")
                    if places_result and len(places_result) > 50:
                        return f"Real-time attractions data: {places_result}"
                except Exception:
                    pass
            
            # Try SerpAPI for fresh Google results
            if self.serp_search:
                try:
                    serp_result = self.serp_search.run(query)
                    if serp_result and len(serp_result) > 50:
                        return f"Latest search results: {serp_result}"
                except Exception:
                    pass
            
            # Try Google Serper
            if self.serper_search:
                try:
                    serper_result = self.serper_search.run(query)
                    if serper_result and len(serper_result) > 50:
                        return f"Current search data: {serper_result}"
                except Exception:
                    pass
            
            # Fallback to DuckDuckGo
            return self.search_tool.invoke(query)
        
        @tool
        def search_restaurants(city: str) -> str:
            """Search for restaurants in a city using real-time data"""
            query = f"best restaurants food places to eat in {city}"
            
            # Try Google Places for real-time restaurant data
            if self.places_tool:
                try:
                    places_result = self.places_tool.run(f"restaurants in {city}")
                    if places_result and len(places_result) > 50:
                        return f"Real-time restaurant data: {places_result}"
                except Exception:
                    pass
            
            # Try SerpAPI for current results
            if self.serp_search:
                try:
                    serp_result = self.serp_search.run(query)
                    if serp_result and len(serp_result) > 50:
                        return f"Latest restaurant results: {serp_result}"
                except Exception:
                    pass
            
            # Fallback to DuckDuckGo
            return self.search_tool.invoke(query)
        
        @tool
        def search_transportation(city: str) -> str:
            """Search for transportation options in a city using real-time data"""
            query = f"transportation options getting around {city} public transport taxi uber"
            
            # Try SerpAPI for current transportation info
            if self.serp_search:
                try:
                    serp_result = self.serp_search.run(query)
                    if serp_result and len(serp_result) > 50:
                        return f"Current transportation info: {serp_result}"
                except Exception:
                    pass
            
            # Try Google Serper
            if self.serper_search:
                try:
                    serper_result = self.serper_search.run(query)
                    if serper_result and len(serper_result) > 50:
                        return f"Latest transport data: {serper_result}"
                except Exception:
                    pass
            
            # Fallback to DuckDuckGo
            return self.search_tool.invoke(query)
        
        @tool
        def get_current_weather(city: str) -> str:
            """Get current weather for a city"""
            weather_data = self.weather_service.get_current_weather(city)
            if weather_data:
                temp = weather_data.get('main', {}).get('temp', 'N/A')
                desc = weather_data.get('weather', [{}])[0].get('description', 'N/A')
                return f"Current weather in {city}: {temp}°C, {desc}"
            return f"Could not fetch weather for {city}"
        
        @tool
        def get_weather_forecast(city: str, days: int = 5) -> str:
            """Get weather forecast for a city"""
            forecast_data = self.weather_service.get_weather_forecast(city, days)
            if forecast_data and 'list' in forecast_data:
                forecast_summary = []
                for i in range(0, min(len(forecast_data['list']), days * 8), 8):
                    item = forecast_data['list'][i]
                    date = item['dt_txt'].split(' ')[0]
                    temp = item['main']['temp']
                    desc = item['weather'][0]['description']
                    forecast_summary.append(f"{date}: {temp}°C, {desc}")
                return f"Weather forecast for {city}:\n" + "\n".join(forecast_summary)
            return f"Could not fetch forecast for {city}"
        
        @tool
        def search_hotels(city: str, budget_range: str = "mid-range") -> str:
            """Search for hotels in a city with budget range using real-time data"""
            query = f"{budget_range} hotels accommodation {city} price per night booking availability"
            
            # Try SerpAPI for real-time hotel prices and availability
            if self.serp_search:
                try:
                    serp_result = self.serp_search.run(query)
                    if serp_result and len(serp_result) > 50:
                        return f"Real-time hotel data: {serp_result}"
                except Exception:
                    pass
            
            # Try Google Places for hotel information
            if self.places_tool:
                try:
                    places_result = self.places_tool.run(f"hotels in {city}")
                    if places_result and len(places_result) > 50:
                        return f"Current hotel listings: {places_result}"
                except Exception:
                    pass
            
            # Try Google Serper
            if self.serper_search:
                try:
                    serper_result = self.serper_search.run(query)
                    if serper_result and len(serp_result) > 50:
                        return f"Latest hotel availability: {serper_result}"
                except Exception:
                    pass
            
            # Fallback to DuckDuckGo
            return self.search_tool.invoke(query)
        
        @tool
        def search_flights(source: str, destination: str) -> str:
            """Search for flight prices from source to destination using real-time data"""
            query = f"flight prices from {source} to {destination}"
            
            # Try SerpAPI for real-time flight prices
            if self.serp_search:
                try:
                    serp_result = self.serp_search.run(query)
                    if serp_result and len(serp_result) > 50:
                        return f"Real-time flight data: {serp_result}"
                except Exception:
                    pass
            
            # Try Google Serper for flight prices
            if self.serper_search:
                try:
                    serper_result = self.serper_search.run(query)
                    if serper_result and len(serp_result) > 50:
                        return f"Current flight data: {serper_result}"
                except Exception:
                    pass
            
            # Fallback to DuckDuckGo
            return self.search_tool.invoke(query)
        
        @tool
        def estimate_hotel_cost(price_per_night: float, total_days: int) -> float:
            """Calculate total hotel cost"""
            return self.calculator.multiply(price_per_night, total_days)
        
        @tool
        def add_costs(cost1: float, cost2: float) -> float:
            """Add two costs together"""
            return self.calculator.add(cost1, cost2)
        
        @tool
        def multiply_costs(cost: float, multiplier: float) -> float:
            """Multiply cost by a multiplier"""
            return self.calculator.multiply(cost, multiplier)
        
        @tool
        def calculate_total_expense(*costs: float) -> float:
            """Calculate total expense from multiple costs"""
            return self.calculator.calculate_total_cost(*costs)
        
        @tool
        def calculate_daily_budget(total_cost: float, days: int) -> float:
            """Calculate daily budget"""
            return self.calculator.calculate_daily_budget(total_cost, days)
        
        @tool
        def get_exchange_rate(from_currency: str, to_currency: str) -> float:
            """Get exchange rate between currencies"""
            return self.currency_service.get_exchange_rate(from_currency, to_currency)
        
        @tool
        def convert_currency(amount: float, from_currency: str, to_currency: str) -> float:
            """Convert amount from one currency to another"""
            return self.currency_service.convert_currency(amount, from_currency, to_currency)
        
        @tool
        def create_day_plan(city: str, day_number: int, attractions: str, weather: str) -> str:
            """Create a day plan for the trip"""
            return f"Day {day_number} in {city}:\n" \
                   f"Weather: {weather}\n" \
                   f"Recommended activities: {attractions[:200]}...\n" \
                   f"Tips: Plan indoor activities if weather is poor."
        
        return [
            search_attractions, search_restaurants, search_transportation,
            get_current_weather, get_weather_forecast, search_hotels, search_flights,
            estimate_hotel_cost, add_costs, multiply_costs, calculate_total_expense,
            calculate_daily_budget, get_exchange_rate, convert_currency, create_day_plan,
            search_images
        ]

class TravelAgent:
    def __init__(self, travel_planner: TravelPlanner):
        self.travel_planner = travel_planner
        self.graph = self._build_graph()

    def get_graph_diagram(self):
        return self.graph.get_graph().draw_mermaid_png()

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(MessagesState)
        workflow.add_node("agent", self.agent_function)
        workflow.add_node("tools", ToolNode(self.travel_planner.tools))
        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges("agent", tools_condition)
        workflow.add_edge("tools", "agent")
        return workflow.compile()

    def agent_function(self, state: MessagesState):
        system_prompt = SystemMessage(content="""You are a helpful AI Travel Agent and Expense Planner. 
            You help users plan trips to any city worldwide with real-time data.
            
            IMPORTANT: Always provide COMPLETE and DETAILED travel plans. Never say "I'll prepare" or "hold on". 
            Give full information immediately including:
            - A day-by-day itinerary with specific times and activities.
            - For each attraction, provide a brief description and an image URL using the search_images tool.
            - For each hotel suggestion, provide a description, price range, and an image URL.
            - A detailed cost breakdown for the entire trip, including flights, hotels, food, and activities.
            - Convert all costs to the user's local currency if specified.
            - Weather details for the duration of the trip.
            
            Format your response in clean Markdown. Use headers, sub-headers, and bullet points to structure the information.
            For images, use the format `![<alt text>](<image url>)`.
            """)
        user_question = state["messages"]
        input_question = [system_prompt] + user_question
        response = self.travel_planner.llm_with_tools.invoke(input_question)
        return {"messages": [response]}

    def plan_trip(self, user_input: str, max_iterations: int = 10) -> str:
        """Main function to plan a trip with iteration control"""
        messages = [HumanMessage(content=user_input)]
        
        # Add iteration counter to prevent infinite loops
        config = {"recursion_limit": max_iterations}
        
        try:
            response = self.graph.invoke({"messages": messages}, config=config)
            final_response = response["messages"][-1].content
            
            # Final check - if still incomplete, force a summary
            if len(final_response) < 800:
                summary_prompt = f"""
                Based on all the information gathered, provide a COMPLETE travel summary now. 
                Don't use tools anymore. Use the information you have to create a comprehensive plan.
                Format your response in clean Markdown with proper headers, lists, and formatting.
                Original request: {user_input}
                """
                
                summary_messages = response["messages"] + [HumanMessage(content=summary_prompt)]
                final_response_obj = self.travel_planner.llm_with_tools.invoke(summary_messages)
                return final_response_obj.content
            
            return final_response
            
        except Exception as e:
            print(f"Workflow error: {e}")
            # Fallback - direct LLM call
            return self._fallback_planning(user_input)

    def _fallback_planning(self, user_input: str) -> str:
        """Fallback method if workflow fails"""
        fallback_prompt = f"""
        Create a complete travel plan for: {user_input}
        
        Provide a comprehensive response including:
        - Daily itinerary
        - Top attractions
        - Restaurant recommendations  
        - Cost estimates
        - Weather information
        - Transportation details
        
        Format your response in clean Markdown with proper headers, lists, and formatting.
        Use your knowledge to provide helpful estimates even without real-time data.
        """
        
        messages = [SystemMessage(content="You are a helpful AI Travel Agent."), HumanMessage(content=fallback_prompt)]
        response = self.travel_planner.llm.invoke(messages)
        return response.content

# --- Streamlit App ---

st.set_page_config(page_title="AI Travel Planner", layout="wide")

st.title("🌍 AI Travel Planner")
st.write("Enter your travel query below and get a detailed itinerary.")

# --- Main App Logic ---
if 'config' not in st.session_state:
    st.session_state.config = Config()

if 'travel_planner' not in st.session_state:
    st.session_state.travel_planner = TravelPlanner(st.session_state.config)

if 'travel_agent' not in st.session_state:
    st.session_state.travel_agent = TravelAgent(st.session_state.travel_planner)

user_query = st.text_input("Your Query:", "Plan a 3-day trip to Paris")

if st.button("Generate Plan"):
    if user_query:
        with st.spinner("Generating your travel plan... This may take a moment."):
            st.markdown("### Travel Plan")
            
            final_plan = st.session_state.travel_agent.plan_trip(user_query)
            
            # Use columns for a better layout
            col1, col2 = st.columns([2, 1])

            with col1:
                st.markdown(final_plan)

            with col2:
                st.subheader("Image Gallery")
                import re
                image_urls = re.findall(r'!\[.*?\]\((.*?)\)', final_plan)
                
                # Create a grid of images
                if image_urls:
                    # Create a grid with 3 columns
                    cols = st.columns(3)
                    for i, url in enumerate(image_urls):
                        with cols[i % 3]:
                            try:
                                st.image(url, use_container_width=True)
                            except Exception as e:
                                st.warning(f"Could not load image: {url} - {e}")

    else:
        st.warning("Please enter a query.")

# --- Sidebar ---
st.sidebar.title("About")
st.sidebar.info(
    "This is an AI-powered travel planner that uses LangChain and Streamlit "
    "to generate detailed travel itineraries based on real-time data."
)
st.sidebar.title("Graph Architecture")
st.sidebar.image(st.session_state.travel_agent.get_graph_diagram())
