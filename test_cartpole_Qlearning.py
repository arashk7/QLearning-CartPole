import gym
from dearpygui import dearpygui as dpg
import numpy as np
import torch
from torch import nn
import pytorch_lightning as pl
import torch.nn.functional as F

class CartPoleModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # Neural network with two layers to approximate Q-function.
        self.network = nn.Sequential(
            nn.Linear(4, 128),  # Input layer for 4-dimensional state space.
            nn.ReLU(),          # Activation function.
            nn.Linear(128, 2)   # Output layer for 2 possible actions.
        )

    def forward(self, x):
        # Forward pass through the network.
        return self.network(x)

    def training_step(self, batch, batch_idx):
        # Training logic for one step.
        state, action, reward, next_state, done = batch
        # Get Q-values for the actions taken.
        current_q_values = self.network(state).gather(1, action.unsqueeze(-1))
        # Calculate expected Q-values from next state.
        next_q_values = self.network(next_state).max(dim=1)[0].detach()
        expected_q_values = reward + (self.gamma * next_q_values * (1 - done))
        # Compute loss.
        loss = F.mse_loss(current_q_values.squeeze(-1), expected_q_values)
        return loss

    def configure_optimizers(self):
        # Optimizer configuration.
        return torch.optim.Adam(self.parameters(), lr=1e-3)
    
# Initialize Dear PyGui.
dpg.create_context()

# Initialize the CartPole environment from OpenAI Gym.
env = gym.make('CartPole-v1')

def draw_cartpole(state, drawlist_id):
    # Visualize the CartPole state using Dear PyGui.
    dpg.delete_item(drawlist_id, children_only=True)
    cart_x, _, pole_angle, _ = state
    # Convert state parameters to pixel values for visualization.
    cart_x_pixel = int(250 + (cart_x * 100)) 
    pole_x_pixel = cart_x_pixel + int(50 * np.sin(pole_angle))
    pole_y_pixel = 200 - int(50 * np.cos(pole_angle))
    # Draw the cart and the pole.
    dpg.draw_rectangle((cart_x_pixel-25, 250), (cart_x_pixel+25, 220), color=(150, 150, 150), fill=(150, 150, 150), parent=drawlist_id)
    dpg.draw_line((cart_x_pixel, 220), (pole_x_pixel, pole_y_pixel), color=(255, 0, 0), thickness=2, parent=drawlist_id)

# Define the window and drawing canvas for visualization.
with dpg.window(label="CartPole Visualization", width=500, height=300):
    drawlist_id = dpg.add_drawlist(width=500, height=300)

def main_loop():
    # Main loop for running the CartPole simulation.
    model = CartPoleModel()  
    state = env.reset()
    while dpg.is_dearpygui_running():
        # Process the state for model input.
        if type(state)==tuple:
            state = state[0]
        state_np = np.array(state).flatten()
        state_tensor = torch.tensor(state_np, dtype=torch.float32).unsqueeze(0)
        # Predict action using the model.
        action = model(state_tensor).argmax().item()
        # Step in the environment using the predicted action.
        state, _, done, _, _ = env.step(action)
        # Update the visualization.
        draw_cartpole(state, drawlist_id)
        if done:
            state = env.reset()
        dpg.render_dearpygui_frame()

# Configure and show the Dear PyGui viewport.
dpg.create_viewport(title='CartPole Visualization', width=600, height=400)
dpg.setup_dearpygui()
dpg.show_viewport()
main_loop()
# Clean up Dear PyGui context.
dpg.destroy_context()
