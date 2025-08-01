import os
import json
import time
import datetime
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Validate API key before creating client
api_key = os.getenv('OPENROUTER_API_KEY')
if not api_key:
    raise ValueError("OPENROUTER_API_KEY environment variable not set. Please set it in your .env file")

# ======================
# Mock Data Generators
# ======================

def get_flights(origin, destination, date):
    return [
        {"airline": "Delta", "flight_number": "DL123", "departure": "08:00 AM", "price": "$350"},
        {"airline": "United", "flight_number": "UA456", "departure": "11:30 AM", "price": "$420"},
        {"airline": "American", "flight_number": "AA789", "departure": "03:15 PM", "price": "$390"}
    ]

def suggest_hotels(destination, check_in, check_out):
    return [
        {"name": "Grand Plaza Hotel", "rating": 4.5, "price": "$120/night", "amenities": ["Pool", "Spa", "Free WiFi"]},
        {"name": "Harbor View Inn", "rating": 4.2, "price": "$95/night", "amenities": ["Beach Access", "Breakfast"]},
        {"name": "City Center Suites", "rating": 4.0, "price": "$110/night", "amenities": ["Gym", "Restaurant"]}
    ]

def get_attractions(destination):
    return [
        {"name": "Historic Downtown", "type": "Cultural", "duration": "3-4 hours"},
        {"name": "Nature Reserve Park", "type": "Outdoor", "duration": "Half day"},
        {"name": "Art Museum", "type": "Indoor", "duration": "2-3 hours"}
    ]

def get_restaurants(destination):
    return [
        {"name": "Seafood Haven", "cuisine": "Seafood", "price_range": "$$$", "rating": 4.7},
        {"name": "Mountain View Bistro", "cuisine": "International", "price_range": "$$", "rating": 4.5},
        {"name": "Local Bites", "cuisine": "Traditional", "price_range": "$", "rating": 4.3}
    ]

# ======================
# Tool Definitions (OpenRouter/OpenAI Format)
# ======================

def get_openai_tools():
    return [
        {
            "type": "function",
            "function": {
                "name": "get_flights",
                "description": "Retrieve available flights between two locations",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "origin": {"type": "string", "description": "Departure city"},
                        "destination": {"type": "string", "description": "Arrival city"},
                        "date": {"type": "string", "description": "Departure date (YYYY-MM-DD)"}
                    },
                    "required": ["origin", "destination", "date"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "suggest_hotels",
                "description": "Find hotels in a destination",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "destination": {"type": "string", "description": "City to search in"},
                        "check_in": {"type": "string", "description": "Check-in date (YYYY-MM-DD)"},
                        "check_out": {"type": "string", "description": "Check-out date (YYYY-MM-DD)"}
                    },
                    "required": ["destination", "check_in", "check_out"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_attractions",
                "description": "Get top attractions in a destination",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "destination": {"type": "string", "description": "City to explore"}
                    },
                    "required": ["destination"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_restaurants",
                "description": "Find recommended restaurants",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "destination": {"type": "string", "description": "City to search in"}
                    },
                    "required": ["destination"]
                }
            }
        }
    ]

# ======================
# Agent Definitions
# ======================
def create_agent(name, instructions, tools):
    """Create an agent definition dictionary"""
    return {
        "name": name,
        "instructions": instructions,
        "tools": tools
    }

# Get tools in proper format
all_tools = get_openai_tools()

# Create specialized agents
destination_agent = create_agent(
    name="Destination Specialist",
    instructions=(
        "Suggest travel destinations based on mood or interests. "
        "Analyze user preferences and recommend exactly ONE suitable location. "
        "Your response should ONLY contain the destination name in the format: 'Destination: [CITY NAME]'."
    ),
    tools=[]
)

booking_agent = create_agent(
    name="Booking Specialist",
    instructions=(
        "Handle travel bookings. Use the following defaults: "
        "Origin city: 'New York', Departure date: 7 days from today, "
        "Return date: 14 days from today. "
        "Use tools to find flights and hotels. "
        "When bookings are complete, say: 'BOOKINGS COMPLETE'."
    ),
    tools=all_tools
)

explore_agent = create_agent(
    name="Exploration Specialist",
    instructions=(
        "Suggest attractions and restaurants. "
        "Use tools to find points of interest and dining options. "
        "Provide comprehensive exploration plans."
    ),
    tools=all_tools
)

# ======================
# Handoff Mechanism
# ======================

def handoff_to_agent(messages, current_agent, next_agent, handoff_message):
    """Transfer conversation to another agent"""
    # Update system message to next agent's instructions
    messages[0] = {"role": "system", "content": next_agent["instructions"]}
    
    # Add handoff message as user message
    messages.append({"role": "user", "content": handoff_message})
    
    return next_agent

# ======================
# Travel Coordinator
# ======================

class TravelDesigner:
    def __init__(self, debug=False):
        self.messages = []
        self.current_agent = destination_agent
        self.user_preferences = {}
        self.debug = debug
        self.max_retries = 5
        self.retry_delay = 30
        # Use a model that supports function calling
        self.model = "anthropic/claude-3-haiku"
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )
        self.full_plan = []  # Store all parts of the travel plan
        
    def log(self, message):
        """Debug logging - only shown if debug=True"""
        if self.debug:
            print(f"[DEBUG] {message}")
            
    def start_conversation(self, user_input):
        self.log(f"Starting conversation with: {user_input}")
        
        # Initialize messages with system prompt and user input
        self.messages = [
            {"role": "system", "content": self.current_agent["instructions"]},
            {"role": "user", "content": user_input}
        ]
        
        return self.process_request()
    
    def handle_rate_limit(self, error, attempt, context):
        """Handle rate limit errors with exponential backoff"""
        if "quota" in error.lower() or "rate" in error.lower() or "credit" in error.lower():
            delay = self.retry_delay * (2 ** attempt)
            self.log(f"Rate limit error during {context}. Waiting {delay} seconds (attempt {attempt+1}/{self.max_retries})...")
            time.sleep(delay)
            return True
        return False
    
    def call_openai(self):
        """Call the OpenRouter API with retry logic"""
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=self.messages,
                    tools=all_tools if self.current_agent["tools"] else None,
                    tool_choice="auto" if self.current_agent["tools"] else None,
                    max_tokens=1024
                )
                return response
            except Exception as e:
                if attempt < self.max_retries - 1:
                    if self.handle_rate_limit(str(e), attempt, "API call"):
                        continue
                self.log(f"API error: {str(e)}")
                raise e
        raise Exception("Max retries exceeded for API call")
    
    def process_tool_calls(self, tool_calls):
        """Execute tool calls and return outputs"""
        tool_outputs = []
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            self.log(f"Calling tool: {function_name} with args: {arguments}")
            
            try:
                if function_name == "get_flights":
                    origin = arguments.get("origin", "New York")
                    destination = arguments.get("destination", "Unknown")
                    date = arguments.get("date", (datetime.date.today() + datetime.timedelta(days=7)).isoformat())
                    output = get_flights(origin, destination, date)
                    
                elif function_name == "suggest_hotels":
                    destination = arguments.get("destination", "Unknown")
                    check_in = arguments.get("check_in", (datetime.date.today() + datetime.timedelta(days=7)).isoformat())
                    check_out = arguments.get("check_out", (datetime.date.today() + datetime.timedelta(days=14)).isoformat())
                    output = suggest_hotels(destination, check_in, check_out)
                    
                elif function_name == "get_attractions":
                    output = get_attractions(arguments.get("destination", "Unknown"))
                    
                elif function_name == "get_restaurants":
                    output = get_restaurants(arguments.get("destination", "Unknown"))
                    
                else:
                    output = "Tool not available"
            except Exception as e:
                output = f"Tool error: {str(e)}"
            
            tool_outputs.append({
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": function_name,
                "content": json.dumps(output)
            })
        
        return tool_outputs
    
    def process_request(self):
        self.log(f"Processing with agent: {self.current_agent['name']}")
        
        while True:
            try:
                # Call API with retry logic
                response = self.call_openai()
                response_message = response.choices[0].message
                self.log(f"API response: {response_message}")
                
                # Handle tool calls if present
                if response_message.tool_calls:
                    tool_calls = response_message.tool_calls
                    
                    # Add assistant message to history
                    self.messages.append({
                        "role": "assistant",
                        "content": None,
                        "tool_calls": tool_calls
                    })
                    
                    # Process tool calls
                    tool_outputs = self.process_tool_calls(tool_calls)
                    
                    # Add tool outputs to messages
                    self.messages.extend(tool_outputs)
                    
                    # Continue processing
                    continue
                    
                # If no tool calls, we have a final response
                assistant_reply = response_message.content
                self.messages.append({
                    "role": "assistant",
                    "content": assistant_reply
                })
                
                # Save agent response to full plan
                self.full_plan.append({
                    "agent": self.current_agent["name"],
                    "response": assistant_reply
                })
                
                # Handle agent handoff logic
                if self.current_agent["name"] == "Destination Specialist":
                    destination = self.extract_destination(assistant_reply)
                    if not destination:
                        return "❌ Error: Could not determine destination"
                        
                    self.user_preferences['destination'] = destination
                    self.log(f"Extracted destination: {destination}")
                    
                    # Calculate default dates
                    today = datetime.date.today()
                    departure_date = (today + datetime.timedelta(days=7)).isoformat()
                    return_date = (today + datetime.timedelta(days=14)).isoformat()
                    
                    handoff_msg = (
                        f"Destination confirmed: {destination}\n"
                        f"Origin: New York\n"
                        f"Departure Date: {departure_date}\n"
                        f"Return Date: {return_date}\n"
                        "Please book flights and hotels."
                    )
                    
                    self.current_agent = handoff_to_agent(
                        self.messages,
                        self.current_agent,
                        booking_agent,
                        handoff_msg
                    )
                    # Continue processing with new agent
                    continue
                    
                elif self.current_agent["name"] == "Booking Specialist":
                    if assistant_reply and "BOOKINGS COMPLETE" in assistant_reply.upper():
                        self.current_agent = handoff_to_agent(
                            self.messages,
                            self.current_agent,
                            explore_agent,
                            "Bookings complete. Please suggest attractions and restaurants."
                        )
                        # Continue processing with new agent
                        continue
                    else:
                        # Return if booking isn't complete
                        return self.format_final_plan()
                        
                else:  # Exploration Specialist
                    # Final agent, return complete plan
                    return self.format_final_plan()
            
            except Exception as e:
                # Handle insufficient credits error specifically
                if "402" in str(e) or "credit" in str(e).lower():
                    return "❌ Error: Insufficient credits on OpenRouter. Please upgrade your account at https://openrouter.ai/settings/credits"
                return f"❌ Error: {str(e)}"
    
    def extract_destination(self, message):
        """Extract destination from agent response"""
        if not message:
            return None
            
        # Try to extract from formatted response
        if "Destination:" in message:
            parts = message.split("Destination:")
            if len(parts) > 1:
                return parts[1].split("\n")[0].strip()
            
        # Try to extract first line if it looks like a location
        lines = [line.strip() for line in message.split('\n') if line.strip()]
        if lines:
            # Check if first line looks like a location name
            if len(lines[0].split()) <= 3 and not any(char.isdigit() for char in lines[0]):
                return lines[0]
                
        # Last resort: return the entire message
        return message
        
    def format_final_plan(self):
        """Format all collected responses into a comprehensive travel plan"""
        if not self.full_plan:
            return "No travel plan generated."
            
        formatted = "✈️ Your Personalized Travel Plan ✈️\n\n"
        
        for part in self.full_plan:
            formatted += f"=== {part['agent']} ===\n"
            formatted += part['response'] + "\n\n"
            
        return formatted

# ======================
# Main Execution
# ======================

def main():
    print("🧳 Welcome to AI Travel Designer!")
    print("Describe your travel preferences (e.g., 'I want a relaxing beach vacation'):")
    user_input = input("> ")
    
    # Create designer with debug mode disabled
    designer = TravelDesigner(debug=False)
    plan = designer.start_conversation(user_input)
    
    print("\n✈️ Here's your personalized travel plan:")
    print(plan)

if __name__ == "__main__":
    main()