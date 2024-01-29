import gym
from dearpygui import dearpygui as dpg
import numpy as np


dpg.create_context()


env = gym.make('CartPole-v1')


def draw_cartpole(state, drawlist_id):
    # Clear previous drawings
    dpg.delete_item(drawlist_id, children_only=True)

    # Extract the cart position and pole angle from the state
    cart_x, _, pole_angle, _ = state

    # Convert these to pixel values
    cart_x_pixel = int(250 + (cart_x * 100))  
    pole_x_pixel = cart_x_pixel + int(50 * np.sin(pole_angle))  
    pole_y_pixel = 200 - int(50 * np.cos(pole_angle))  

    # Draw the cart (as a rectangle)
    dpg.draw_rectangle((cart_x_pixel-25, 250), (cart_x_pixel+25, 220), color=(150, 150, 150), fill=(150, 150, 150), parent=drawlist_id)

    # Draw the pole (as a line)
    dpg.draw_line((cart_x_pixel, 220), (pole_x_pixel, pole_y_pixel), color=(255, 0, 0), thickness=2, parent=drawlist_id)


with dpg.window(label="CartPole Visualization", width=500, height=300):
    drawlist_id = dpg.add_drawlist(width=500, height=300)

# Main loop
def main_loop():
    state = env.reset()
    while dpg.is_dearpygui_running():
        action = env.action_space.sample() 
        state, _, done, _,_ = env.step(action)

        draw_cartpole(state, drawlist_id)

        if done:
            state = env.reset()

        dpg.render_dearpygui_frame()

# Create Dear PyGui viewport
dpg.create_viewport(title='CartPole Visualization', width=600, height=400)
dpg.setup_dearpygui()
dpg.show_viewport()


main_loop()


dpg.destroy_context()
