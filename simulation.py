import matplotlib.pyplot as plt
from thermosnooker.multiball_simulation import MultiBallSimulation

def run_simulation():
    mbs = MultiBallSimulation(rmax=8, nrings=3, multi=3)
    mbs.run(500, pause_time=0.01, animate=True)

if __name__ == "__main__":
    run_simulation()
