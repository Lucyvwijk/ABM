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

# Visualize time series
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Energy Label Transition Model - 30 Year Simulation', 
             fontsize=16, fontweight='bold')

# Plot 1: Cumulative transitions
axes[0].plot(model_data.index, model_data['Num_Transitioned'], 
             linewidth=2.5, color='green', marker='o', markersize=4)
axes[0].set_xlabel('Year', fontsize=11)
axes[0].set_ylabel('Number of Transitions', fontsize=11)
axes[0].set_title('Cumulative Transitions', fontsize=12, fontweight='bold')
axes[0].grid(True, alpha=0.3)

# Plot 2: Average label vs target
axes[1].plot(model_data.index, model_data['Avg_Label_Numeric'], 
             linewidth=2.5, color='blue', label='Actual')
target_value = label_to_numeric(model.target_label)
axes[1].axhline(y=target_value, color='red', linestyle='--', 
                linewidth=2, label=f'Target: {model.target_label}')
axes[1].set_xlabel('Year', fontsize=11)
axes[1].set_ylabel('Average Label', fontsize=11)
axes[1].set_title('City Average Energy Label', fontsize=12, fontweight='bold')
axes[1].set_yticks(range(8))
axes[1].set_yticklabels(['G', 'F', 'E', 'D', 'C', 'B', 'A', 'A+'])
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import time
import matplotlib

# Increase the limit for embedding animations in notebooks
matplotlib.rcParams['animation.embed_limit'] = 2**128

start_time = time.time()

# Create a new model to run and animate
print("Creating new model for animation...")
animation_model = City(
    num_houses=50,
    num_owners=40,
    grid_size=10,
    financial_status_mean=30000,
    financial_status_std=5000,
    target_label="A",
    seed=42
)

# Store states for all 30 years (including year 0)
print("Running simulation and collecting grid states for 30 years...")
grid_states = []
stats_over_time = []

# Collect initial state (year 0)
# Use Mesa's agents_by_type to filter houses
houses = list(animation_model.agents_by_type[House])
# Create empty grid filled with -1 (represents empty cells)
grid_viz = np.full((animation_model.grid_size, animation_model.grid_size), -1.0)

# Fill grid with house information
for house in houses:
    x, y = house.pos
    if house.owner_id is None:
        grid_viz[y, x] = -0.5  # Vacant house (no owner)
    else:
        grid_viz[y, x] = label_to_numeric(house.current_label)  # Store label as number

# Store initial grid state and statistics
grid_states.append(grid_viz.copy())
stats_over_time.append({
    'year': 0,
    'transitions': animation_model.count_transitions(),
    'avg_label': animation_model.get_current_avg_label()
})

# Run simulation and collect states for each year
for year in range(1, 31):
    animation_model.step()  # Advance simulation one year
    
    # Update grid state for this year
    # Use Mesa's agents_by_type for filtering
    houses = list(animation_model.agents_by_type[House])
    grid_viz = np.full((animation_model.grid_size, animation_model.grid_size), -1.0)
    
    # Fill grid with current house information
    for house in houses:
        x, y = house.pos
        if house.owner_id is None:
            grid_viz[y, x] = -0.5  # Vacant house
        else:
            grid_viz[y, x] = label_to_numeric(house.current_label)  # Current label
    
    # Store this year's grid state and statistics
    grid_states.append(grid_viz.copy())
    stats_over_time.append({
        'year': year,
        'transitions': animation_model.count_transitions(),
        'avg_label': animation_model.get_current_avg_label()
    })
    
    # Print progress every 10 years
    if year % 10 == 0:
        print(f"  Collected state for year {year}")

print(f"Collected {len(grid_states)} grid states")
print("\nCreating animation (this may take 1-2 minutes)...")

# Create figure for animation
fig, ax = plt.subplots(figsize=(12, 10))
plt.tight_layout(rect=[0, 0, 0.85, 1])

# Define colors and colormap
# White=empty, LightGray=vacant, then G(red) → F → E → D → C → B → A → A+(blue)
colors = ['white', 'lightgray', '#d62728', '#ff7f0e', '#ffbb78', '#ffd700', 
          '#90ee90', '#2ca02c', '#1f77b4', '#0000ff']
cmap = ListedColormap(colors)

# Create legend elements
legend_elements = [
    Patch(facecolor='white', edgecolor='black', label='Empty Cell'),
    Patch(facecolor='lightgray', edgecolor='black', label='Vacant House'),
    Patch(facecolor='#d62728', edgecolor='black', label='Label G'),
    Patch(facecolor='#ff7f0e', edgecolor='black', label='Label F'),
    Patch(facecolor='#ffbb78', edgecolor='black', label='Label E'),
    Patch(facecolor='#ffd700', edgecolor='black', label='Label D'),
    Patch(facecolor='#90ee90', edgecolor='black', label='Label C'),
    Patch(facecolor='#2ca02c', edgecolor='black', label='Label B'),
    Patch(facecolor='#1f77b4', edgecolor='black', label='Label A'),
    Patch(facecolor='#0000ff', edgecolor='black', label='Label A+')
]

# Function to plot grid at each frame of animation
def plot_grid(frame):
    ax.clear()  # Clear previous frame
    
    # Get grid state and stats for this frame (year)
    grid_data = grid_states[frame]
    stats = stats_over_time[frame]
    
    # Map values for colormap: -1→0 (empty), -0.5→1 (vacant), 0-7→2-9 (labels G-A+)
    display_grid = np.where(grid_data == -1, 0,
                   np.where(grid_data == -0.5, 1, grid_data + 2))
    
    # Display grid with colors
    im = ax.imshow(display_grid, cmap=cmap, vmin=0, vmax=9, origin='lower')
    
    # Add grid lines between cells
    ax.set_xticks(np.arange(-0.5, animation_model.grid_size, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, animation_model.grid_size, 1), minor=True)
    ax.grid(which="minor", color="black", linestyle='-', linewidth=0.5)
    ax.set_xticks([])  # Hide axis tick labels
    ax.set_yticks([])
    
    # Add legend showing what each color means
    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.02, 0.5), 
              fontsize=9, frameon=True)
    
    # Title with year and statistics
    ax.set_title(
        f'Energy Label Transition Model - Year {stats["year"]}\n'
        f'Transitions: {stats["transitions"]}/{animation_model.num_owners} | '
        f'Avg Label: {stats["avg_label"]}',
        fontsize=14, fontweight='bold', pad=15
    )

# Initial plot (frame 0)
plot_grid(0)

# Update function for animation - called for each frame
def update(frame):
    plot_grid(frame)

# Create animation (31 frames for years 0-30)
anim = FuncAnimation(fig, update, frames=31, repeat=True, interval=300)

# Convert to HTML for display in notebook
output = HTML(anim.to_jshtml())

end_time = time.time()
elapsed_time = end_time - start_time

print(f"\n✓ Animation complete!")
print(f"  Time taken: {elapsed_time:.1f} seconds")
print(f"  Total frames: 31 (years 0-30)")
print(f"  Frame interval: 300ms (~3 frames per second)")

# Display the animation
output


