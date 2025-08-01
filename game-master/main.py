import os
import json
import random
from openai import OpenAI
from dotenv import load_dotenv
from termcolor import colored

# Load environment variables
load_dotenv()

class GameState:
    """Maintains the game state across agents"""
    def __init__(self):
        self.player_health = 100
        self.player_inventory = []
        self.current_location = "Forest Entrance"
        self.game_over = False
        self.combat_active = False
        self.monster = None
        self.item_discovered = None
        self.last_action = ""

# Initialize OpenAI client with OpenRouter configuration
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    default_headers={
        "HTTP-Referer": "http://localhost:3000",  # Optional for identifying your app
        "X-Title": "Fantasy Adventure Game"       # Optional for identifying your app
    }
)

# ------------------------
# Tool Implementations
# ------------------------
def roll_dice(**kwargs):
    """Rolls dice for game mechanics"""
    sides = kwargs.get('sides', 6)
    count = kwargs.get('count', 1)
    results = [random.randint(1, sides) for _ in range(count)]
    return f"Dice rolls: {results} (Total: {sum(results)})"

def generate_event(**kwargs):
    """Generates random game events"""
    events = [
        "A mysterious glowing object appears on the path",
        "A hidden trap triggers!",
        "Strange whispers echo from the trees",
        "The path forks in two directions",
        "A sudden fog rolls in, reducing visibility",
        "Ancient runes carved into a stone glow faintly"
    ]
    return random.choice(events)

# Define tool specifications
roll_dice_tool = {
    "type": "function",
    "function": {
        "name": "roll_dice",
        "description": "Roll dice for game mechanics",
        "parameters": {
            "type": "object",
            "properties": {
                "sides": {"type": "integer", "description": "Number of sides per die"},
                "count": {"type": "integer", "description": "Number of dice to roll"}
            },
            "required": []
        }
    }
}

generate_event_tool = {
    "type": "function",
    "function": {
        "name": "generate_event",
        "description": "Generate random game events",
        "parameters": {"type": "object", "properties": {}}
    }
}

# ------------------------
# Agent Definitions
# ------------------------
class BaseAgent:
    """Base class for all agents with common functionality"""
    def __init__(self, state, system_message, tools=None):
        self.state = state
        self.system_message = system_message
        self.tools = tools or []
    
    def generate_response(self, player_input):
        """Generate a response using the ChatCompletion API"""
        # Create message with game state
        state_info = self._get_state_info()
        full_input = f"{state_info}\n\nPlayer: {player_input}"
        
        messages = [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": full_input}
        ]
        
        # Process tool calls recursively
        return self._process_tool_calls(messages)
    
    def _process_tool_calls(self, messages):
        """Process tool calls recursively until no more tool calls are needed"""
        while True:
            response = client.chat.completions.create(
                model="google/gemini-2.0-flash-001",  # OpenRouter model specification
                messages=messages,
                tools=self.tools if self.tools else None
            )
            
            message = response.choices[0].message
            if not message.tool_calls:
                return message.content
            
            # Add assistant message with tool calls to conversation
            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": message.tool_calls
            })
            
            # Process each tool call
            for tool_call in message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                
                # Call the appropriate function
                if function_name == "roll_dice":
                    result = roll_dice(**function_args)
                elif function_name == "generate_event":
                    result = generate_event(**function_args)
                else:
                    result = f"Error: Unknown function {function_name}"
                
                # Add tool response to conversation
                messages.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": result
                })
    
    def _get_state_info(self):
        """Get state information (to be implemented by subclasses)"""
        raise NotImplementedError

class NarratorAgent(BaseAgent):
    """Handles story narration and game progression"""
    def __init__(self, state):
        system_message = (
            "You are the narrator of a fantasy adventure game. Describe the environment, progress the story, "
            "and respond to player actions. Use tools for game mechanics. When combat starts, output exactly: "
            "'HANDOFF_MONSTER: [monster name]'. When an item is discovered, output exactly: 'HANDOFF_ITEM: [item name]'."
        )
        super().__init__(state, system_message, [roll_dice_tool, generate_event_tool])
    
    def _get_state_info(self):
        return (
            f"Location: {self.state.current_location}\n"
            f"Health: {self.state.player_health} HP\n"
            f"Inventory: {', '.join(self.state.player_inventory) if self.state.player_inventory else 'Empty'}\n"
            f"Last Action: {self.state.last_action}"
        )

class MonsterAgent(BaseAgent):
    """Handles combat encounters"""
    def __init__(self, state):
        system_message = (
            "You handle combat encounters. Describe monster actions, calculate damage using dice rolls, "
            "and manage combat mechanics. Hand back to NarratorAgent when combat ends by outputting exactly: "
            "'HANDOFF_NARRATOR'. Use dice roll tool for combat mechanics."
        )
        super().__init__(state, system_message, [roll_dice_tool])
    
    def _get_state_info(self):
        return (
            f"Combat with: {self.state.monster}\n"
            f"Player Health: {self.state.player_health} HP\n"
            f"Last Action: {self.state.last_action}"
        )

class ItemAgent(BaseAgent):
    """Handles item discovery and inventory management"""
    def __init__(self, state):
        system_message = (
            "You handle item discovery and inventory management. Describe discovered items, manage player inventory. "
            "Hand back to NarratorAgent when done by outputting exactly: 'HANDOFF_NARRATOR'."
        )
        super().__init__(state, system_message)
    
    def _get_state_info(self):
        return (
            f"Discovered Item: {self.state.item_discovered}\n"
            f"Player Inventory: {', '.join(self.state.player_inventory) if self.state.player_inventory else 'Empty'}"
        )

# ------------------------
# Game Master
# ------------------------
class GameMaster:
    """Orchestrates the game flow between agents"""
    def __init__(self):
        self.state = GameState()
        self.narrator = NarratorAgent(self.state)
        self.monster_agent = MonsterAgent(self.state)
        self.item_agent = ItemAgent(self.state)
        self.current_agent = self.narrator
        self.agent_name = "Narrator"
    
    def display_status(self):
        """Display current game status"""
        print(colored("\n=== ADVENTURE STATUS ===", "cyan"))
        print(f"Health: {self.state.player_health} HP")
        print(f"Inventory: {', '.join(self.state.player_inventory) if self.state.player_inventory else 'Empty'}")
        print(f"Location: {self.state.current_location}")
        
        if self.state.combat_active:
            print(f"Combat with: {self.state.monster}")
        elif self.state.item_discovered:
            print(f"Discovered: {self.state.item_discovered}")
        
        print(colored("========================", "cyan"))
    
    def process_response(self, response):
        """Process agent response and handle special commands"""
        # Check for handoff commands
        if "HANDOFF_MONSTER:" in response:
            parts = response.split("HANDOFF_MONSTER:")
            self.state.monster = parts[1].strip()
            self.state.combat_active = True
            self.current_agent = self.monster_agent
            self.agent_name = "Monster"
            return parts[0].strip()
        
        if "HANDOFF_ITEM:" in response:
            parts = response.split("HANDOFF_ITEM:")
            self.state.item_discovered = parts[1].strip()
            self.current_agent = self.item_agent
            self.agent_name = "Item"
            return parts[0].strip()
        
        if "HANDOFF_NARRATOR" in response:
            parts = response.split("HANDOFF_NARRATOR")
            self.state.combat_active = False
            self.state.monster = None
            self.state.item_discovered = None
            self.current_agent = self.narrator
            self.agent_name = "Narrator"
            return parts[0].strip() if len(parts) > 1 else response
        
        # Check for game state updates
        if "UPDATE_LOCATION:" in response:
            parts = response.split("UPDATE_LOCATION:")
            self.state.current_location = parts[1].strip()
            return parts[0].strip()
        
        if "DAMAGE_PLAYER:" in response:
            parts = response.split("DAMAGE_PLAYER:")
            damage = int(parts[1].strip())
            self.state.player_health -= damage
            if self.state.player_health <= 0:
                self.state.player_health = 0
                self.state.game_over = True
            return parts[0].strip()
        
        if "HEAL_PLAYER:" in response:
            parts = response.split("HEAL_PLAYER:")
            heal = int(parts[1].strip())
            self.state.player_health += heal
            return parts[0].strip()
        
        if "ADD_ITEM:" in response:
            parts = response.split("ADD_ITEM:")
            self.state.player_inventory.append(parts[1].strip())
            return parts[0].strip()
        
        return response
    
    def start_game(self):
        """Start the game loop"""
        print(colored("Welcome to the Fantasy Adventure Game!", "green", attrs=["bold"]))
        print(colored("Type 'quit' at any time to exit the game\n", "yellow"))
        
        # Initial narration
        response = self.narrator.generate_response("Begin the adventure")
        print(colored(f"\nNarrator: {response}\n", "yellow"))
        
        while not self.state.game_over:
            self.display_status()
            
            # Get user input
            user_input = input(colored(f"{self.agent_name} > ", "green")).strip()
            
            if user_input.lower() == "quit":
                print("Thanks for playing!")
                break
            
            # Store last action
            self.state.last_action = user_input
            
            # Get agent response
            response = self.current_agent.generate_response(user_input)
            
            # Process special commands in response
            processed_response = self.process_response(response)
            
            # Display response
            print(colored(f"\n{self.agent_name}: {processed_response}\n", "yellow"))
            
            # Check for player defeat
            if self.state.player_health <= 0:
                print(colored("Game Over! You have been defeated.", "red"))
                break

if __name__ == "__main__":
    game = GameMaster()
    game.start_game()