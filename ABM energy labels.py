# Core Mesa library for agent-based modeling
import mesa

# Numerical and data manipulation
import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

# Type hints for better code documentation
from typing import Dict, List, Tuple

# Print Mesa version to confirm correct installation
print(f"Mesa version: {mesa.__version__}")
print("Required: Mesa 3.0.3 or higher")   

def label_to_numeric(label: str) -> int:
    """Convert energy label to numeric value (G=0 to A+=7)."""
    # Dictionary mapping letter labels to numbers for calculations
    mapping = {"G": 0, "F": 1, "E": 2, "D": 3, "C": 4, "B": 5, "A": 6, "A+": 7}
    return mapping.get(label, 0)  # Return 0 if label not found


def numeric_to_label(value: float) -> str:
    """Convert numeric value back to energy label string."""
    # Dictionary mapping numbers back to letter labels
    mapping = {0: "G", 1: "F", 2: "E", 3: "D", 4: "C", 5: "B", 6: "A", 7: "A+"}
    rounded = int(round(value))  # Round to nearest integer
    rounded = max(0, min(7, rounded))  # Clamp to valid range [0, 7]
    return mapping[rounded] # This returns the corresponding label


def calculate_transition_cost(current_label: str, target_label: str) -> float:
    """Calculate cost to transition from current to target label.
    
    Simple linear cost model: 10,000€ per label step improvement.
    Example: D→B costs 20,000€ (2 steps × 10,000€)
    """
    # Calculate how many label steps need to be improved
    gap = label_to_numeric(target_label) - label_to_numeric(current_label)
    
    # If already at or above target, no cost needed
    if gap <= 0:
        return 0.0
    
    # Linear cost: €10,000 per label step
    return gap * 10000

class House(mesa.Agent):
    """House agent - passive container for energy efficiency information."""

    def __init__(self, model, current_label: str):
        super().__init__(model)  # Initialize Mesa Agent. Note that mesa will assign a unique ID to each agent instance
        self.current_label = current_label  # Energy label (A+ to G)
        self.owner_id = None  # ID of owner living here (None if vacant)
    
    def step(self):
        """Houses are passive - no actions needed each time step."""
        pass

class Owner(mesa.Agent):
    """Homeowner who makes transition decisions."""
    
    def __init__(self, model, financial_status: float):
        super().__init__(model)  
        self.financial_status = financial_status  # Available budget (€)
        self.house = None  
        self.has_transitioned = False  # Track if already upgraded (can only do once)
    
    def step(self):
        """Called each year - owner decides whether to transition."""
        # Only consider transitioning if haven't already done so AND have a house
        if not self.has_transitioned and self.house is not None: # Note that each owner has a house assigned, but this can provide a safeguard just in case
            # Calculate cost to reach model's target label
            cost = calculate_transition_cost(
                self.house.current_label,
                self.model.target_label
            )
            
            # Simple affordability check: can I afford this transition?
            if cost > 0 and self.financial_status >= cost:
                # Perform the transition
                self.house.current_label = self.model.target_label
                self.has_transitioned = True  # Mark as transitioned

class City(mesa.Model):
    """City model with houses and owners on a grid."""
    
    def __init__(
        self,
        num_houses: int = 50, # Total number of houses in the model
        num_owners: int = 40, # Total number of owners in the model
        grid_size: int = 10, # Size of the grid (grid_size x grid_size). Note that you can define the width and height separately with different dimensions if desired
        financial_status_mean: float = 30000, # Average financial status of owners (€). This helps controlling the affordability of transitions in the base model
        financial_status_std: float = 5000, # Standard deviation of financial status (€). This helps controlling the affordability of transitions in the base model
        label_distribution: Dict[str, float] = None, # Distribution of energy labels among houses. This allows customizing the initial state of the housing stock
        target_label: str = "A", # Target energy label for transitions
        seed: int = None
    ):
        # Initialize Mesa Model with seed (Mesa 3.0 handles random number generation)
        super().__init__(seed=seed)
        
        # Validate inputs. Return an error if the number of owners is greater than the number of houses
        if num_owners > num_houses:
            raise ValueError(f"Cannot have more owners ({num_owners}) than houses ({num_houses})!")
        
        # Assign parameters to model attributes
        self.num_houses = num_houses
        self.num_owners = num_owners
        self.grid_size = grid_size
        self.financial_status_mean = financial_status_mean
        self.financial_status_std = financial_status_std
        self.target_label = target_label
        
        # Default label distribution (more poor labels, fewer good ones). If this line is confusing, please ask teh instructor.
        if label_distribution is None:
            self.label_distribution = {
                "G": 0.05, "F": 0.10, "E": 0.20, "D": 0.30,
                "C": 0.20, "B": 0.10, "A": 0.04, "A+": 0.01
            }
        else:
            self.label_distribution = label_distribution
        
        # Initialize grid using Mesa's MultiGrid object (please read the documentation for more details)
        self.grid = mesa.space.MultiGrid(grid_size, grid_size, torus=False)
        
        # Create agents
        self.create_houses()
        self.create_owners()

        # Assign houses to owners based on their grid positions
        self.assign_houses_to_owners()
        
        # Set up data collection
        self.setup_data_collection()
        self.datacollector.collect(self)
    
    def create_houses(self):
        """Create and place houses on grid."""
        # Extract label names and probabilities from distribution
        labels = list(self.label_distribution.keys())
        probs = list(self.label_distribution.values())
        
        # Check if we have enough grid cells for all houses
        total_cells = self.grid_size * self.grid_size
        if self.num_houses > total_cells: #raise error if not enough space
            raise ValueError(f"Cannot place {self.num_houses} houses on a {self.grid_size}×{self.grid_size} grid (only {total_cells} cells available)!")
        
        # Generate all possible grid positions
        all_positions = [(x, y) for x in range(self.grid_size) for y in range(self.grid_size)]
        
        # Sample unique positions without replacement (much more efficient than while loop)
        selected_positions = self.random.sample(all_positions, self.num_houses)
        
        # Create houses and place them at the selected positions
        for pos in selected_positions:
            # Randomly select initial label based on distribution
            label = self.random.choices(labels, weights=probs, k=1)[0] # this line means that we are selecting one label based on the weights provided in probs
            house = House(self, label)
            self.grid.place_agent(house, pos) # Mesa's method for placing agents on a grid
    
    def create_owners(self):
        """Create and place owners at house positions."""
        # Use Mesa's AgentSet.get() to extract positions. Refer to Mesa 3.0.3 documentation to learn more about AgentSet and filtering
        house_positions = self.agents_by_type[House].get("pos") # Note that agents_by_type returns a list
        
        # Ensure we don't try to place more owners than available houses. Note that this is redundant given the earlier check of num_owners being less or equal to num_houses, but added here for safety
        if self.num_owners > len(house_positions):
            raise ValueError(f"Cannot place {self.num_owners} owners in {len(house_positions)} houses!")
        
        # Sample unique positions without replacement to ensure each owner gets a different house
        selected_positions = self.random.sample(house_positions, self.num_owners)
        
        # Create owners agents and assign each to a unique house position
        for pos in selected_positions:
            # Draw financial status from normal distribution. This help us ensure that the mean that we define is respected while allowing for variability
            financial_status = max(0, self.random.gauss(
                self.financial_status_mean,
                self.financial_status_std
            ))  # Ensure non-negative
            
            # create owner agent instance using the drawn financial status
            owner = Owner(self, financial_status)
            
            # Place owner at the selected house position
            self.grid.place_agent(owner, pos)
    
    def assign_houses_to_owners(self):
        """Assign houses to owners based on their grid positions.
        
        Since owners are placed at house positions, we just find the house at the same cell.
        """
        owners = self.agents_by_type[Owner]
        
        for owner in owners:
            # Get all agents at this cell - should be exactly one house and one owner
            cellmates = self.grid.get_cell_list_contents([owner.pos])
            
            # Safety check: ensure exactly 2 agents (house + owner) at this position
            if len(cellmates) != 2:
                raise ValueError(f"Expected 2 agents at position {owner.pos}, found {len(cellmates)}")
            
            # Find the house (the agent that is not the owner)
            # cellmates[0] and cellmates[1] are the two agents - one is owner, other is house
            house = cellmates[0] if cellmates[0] is not owner else cellmates[1]
            
            # Link them together
            owner.house = house
            house.owner_id = owner.unique_id
    
    def setup_data_collection(self):
        """Configure data collection using Mesa's DataCollector."""
        self.datacollector = mesa.DataCollector(
            # Model-level metrics (one value per time step)
            model_reporters={
                "Num_Transitioned": self.count_transitions,  # Direct method reference since the method is defined inside the model class
                "Avg_Label_Numeric": lambda m: label_to_numeric(m.get_current_avg_label()), #Lambda is used when you need to do an operation beyond just passing a ready-to-use function or an attribute. Please look up lambda function if you are unfamiliar with it. Ask your instructor if you need more clarification
                "Avg_Label": self.get_current_avg_label
            },
            # Agent-level metrics (one value per agent per time step)
            agent_reporters={
                "Agent_Type": lambda a: type(a).__name__,  # "Owner" or "House"
                "Financial_Status": lambda a: getattr(a, 'financial_status', None), # This means that for each agent (owners and houses), check if they have financial_status attribute. If yes, then report its value; if not, write None. 
                                                                                    # Remember that we model houses as agents without financial status so we need to handle that
                "Has_Transitioned": lambda a: getattr(a, 'has_transitioned', None),
                "House_Label": lambda a: a.house.current_label if hasattr(a, 'house') and a.house else None
            }
        )
    
    def get_current_avg_label(self) -> str:
        """Calculate average energy label across occupied houses."""
        # Use Mesa's agents_by_type and AgentSet.select(). Please check Mesa 3.0.3 documentation to learn more about AgentSet and filtering
        occupied = self.agents_by_type[House].select(lambda h: h.owner_id is not None) # Stare a bit at this line to understand how it works! a lot is happening in this line
        
        # Return default if no occupied houses
        if len(occupied) == 0:
            return "D"
        
        # Convert labels to numbers, calculate average, convert back to label
        total = sum(label_to_numeric(h.current_label) for h in occupied)
        avg = total / len(occupied)
        return numeric_to_label(avg)
    
    def count_transitions(self) -> int:
        """Count owners who have transitioned."""
        # Use Mesa's agents_by_type to get all owners
        owners = self.agents_by_type[Owner] 
        return sum(1 for o in owners if o.has_transitioned)
    
    def step(self):
        """Execute one simulation step (one year)."""
        # Use Mesa's AgentSet.do() to call step() on only owners
        # More efficient than calling step on all agents since houses are passive. 
        self.agents_by_type[Owner].do("step") # If you want to step all the agents, you can use self.agents.do("step")
        # Collect data for this time step
        self.datacollector.collect(self)

# Create and run model
print("Creating model...")
model = City(
    num_houses=50,
    num_owners=40,
    grid_size=10,
    financial_status_mean=30000,
    financial_status_std=5000,
    target_label="A",
    seed=42
) # note that you can decide to not define the values for the parameters. In this case, default values defined in the model class will be used

# Store initial label counts for comparison later
initial_label_counts = {}
for house in model.agents_by_type[House]:
    label = house.current_label
    initial_label_counts[label] = initial_label_counts.get(label, 0) + 1

print("\n" + "="*60)
print("INITIAL STATE (Year 0)")
print("="*60)
print(f"Number of houses:       {model.num_houses}")
print(f"Number of owners:       {model.num_owners}")
print(f"Grid size:              {model.grid_size} × {model.grid_size}")
print(f"Initial avg label:      {model.get_current_avg_label()}")
print(f"Target label:           {model.target_label}")
print("="*60)

# Run simulation
print("\nRunning simulation for 30 years...")
num_years = 30

for year in range(num_years):
    model.step()
    if (year + 1) % 10 == 0:
        print(f"  Year {year + 1}: Avg label = {model.get_current_avg_label()}, "
              f"Transitions = {model.count_transitions()}")

print("\n" + "="*60)
print(f"FINAL STATE (Year {num_years})")
print("="*60)
print(f"Final avg label:        {model.get_current_avg_label()}")
print(f"Total transitions:      {model.count_transitions()} / {model.num_owners} "
      f"({model.count_transitions()/model.num_owners*100:.1f}%)")
print("="*60)

# Get model-level data (one row per year)
model_data = model.datacollector.get_model_vars_dataframe()
model_data

# Get agent-level data (one row per owner per year)
agent_data = model.datacollector.get_agent_vars_dataframe()
agent_data

