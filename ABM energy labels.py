# Import necessary libraries
import random
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector

# Define the Homeowner Agent
class Homeowner(Agent):
    """ An agent representing a homeowner with a budget, willingness, and knowledge about sustainability. """
    
    def __init__(self, unique_id, model, budget, willingness, knowledge):
        super().__init__(unique_id, model)
        self.budget = budget
        self.willingness = willingness
        self.knowledge = knowledge
        self.energy_label = "D"  # Initial energy label (for example)
    
    def step(self):
        # Step 1: Information campaign effect on willingness
        if self.model.info_campaign:
            self.willingness += self.model.info_campaign_effect
        
        # Step 2: Subsidy effect on budget
        if self.model.subsidy_campaign:
            self.budget += self.model.subsidy_amount
        
        # Step 3: Check if the homeowner is willing and has enough budget to improve their energy label
        if self.willingness > random.random() and self.budget >= self.model.improvement_cost:
            # Improve the energy label
            self.energy_label = self.model.improve_energy_label(self.energy_label)

        # Step 4: Decrease knowledge or willingness over time
        self.willingness = max(self.willingness - 0.05, 0)  # Decay in willingness
        self.knowledge = max(self.knowledge - 0.01, 0)  # Decay in knowledge

# Define the Model
class EnergyLabelModel(Model):
    """A model to simulate homeowners improving energy labels based on government policies."""
    
    def __init__(self, num_agents, width, height, info_campaign, subsidy_campaign, info_campaign_effect, subsidy_amount, improvement_cost):
        self.num_agents = num_agents
        self.grid = MultiGrid(width, height, True)
        self.schedule = RandomActivation(self)
        self.info_campaign = info_campaign
        self.subsidy_campaign = subsidy_campaign
        self.info_campaign_effect = info_campaign_effect
        self.subsidy_amount = subsidy_amount
        self.improvement_cost = improvement_cost

        # Create the agents
        for i in range(self.num_agents):
            budget = random.randint(5000, 20000)  # Budget between 5k and 20k
            willingness = random.random()  # Willingness between 0 and 1
            knowledge = random.random()  # Knowledge between 0 and 1
            a = Homeowner(i, self, budget, willingness, knowledge)
            self.schedule.add(a)

            # Add the agent to a random grid cell
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(a, (x, y))
        
        # Data collector to track the progress
        self.datacollector = DataCollector(
            agent_reporters={"Energy Label": "energy_label"}
        )

    def improve_energy_label(self, current_label):
        """ Improve the energy label based on the current label. """
        label_order = ['D', 'C', 'B', 'A']
        next_label = label_order[label_order.index(current_label) + 1] if label_order.index(current_label) < len(label_order) - 1 else current_label
        return next_label

    def step(self):
        """Advance the model by one step."""
        self.datacollector.collect(self)
        self.schedule.step()

# Create the model and run it
model = EnergyLabelModel(
    num_agents=100, 
    width=10, 
    height=10, 
    info_campaign=True, 
    subsidy_campaign=True, 
    info_campaign_effect=0.1,  # 10% increase in willingness
    subsidy_amount=2000,  # Subsidy of 2000 to each agent
    improvement_cost=1000  # Cost to improve energy label
)

# Run the model for 100 steps
for i in range(100):
    model.step()

# Collect data
agent_data = model.datacollector.get_agent_vars_dataframe()
print(agent_data.tail())  # Output the last step data for inspection
