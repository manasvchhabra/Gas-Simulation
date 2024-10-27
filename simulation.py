import tkinter as tk
from tkinter import simpledialog
import matplotlib.pyplot as plt
from thermosnooker.simulations import MultiBallSimulation

def run_simulation(num_collisions):
    mbs = MultiBallSimulation(rmax=8, nrings=3, multi=3)
    mbs.run(num_collisions, pause_time=0.001, animate=True)

def get_user_input():
    # Create a simple Tkinter window
    root = tk.Tk()
    root.withdraw()  # Hide the root window

    # Ask the user for the number of collisions
    num_collisions = simpledialog.askinteger("Input", "Enter the number of collisions:", minvalue=1)

    if num_collisions:
        run_simulation(num_collisions)

if __name__ == "__main__":
    get_user_input()
