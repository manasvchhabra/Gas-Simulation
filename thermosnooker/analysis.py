"""Analysis Module."""
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

try:
    from balls import Ball, Container
    from simulations import SingleBallSimulation, MultiBallSimulation
    from physics import maxwell
except ImportError:
    from .balls import Ball, Container
    from .simulations import SingleBallSimulation, MultiBallSimulation
    from .physics import maxwell
def task9():
    """
    Task 9.

    In this function, you should test your animation. To do this, create a container
    and ball as directed in the project brief. Create a SingleBallSimulation object from these
    and try running your animation. Ensure that this function returns the balls final position and
    velocity.

    Returns:
        tuple[NDArray[np.float64], NDArray[np.float64]]: The balls final position and velocity
    """
    cont = Container(radius=10.)
    ball = Ball(pos=[-5, 0], vel=[1, 0.], radius=1., mass=1.)
    sbs = SingleBallSimulation(container=cont, ball=ball)
    sbs.run(20, pause_time=0.5, animate=True)
    return (sbs.ball().pos(), sbs.ball().vel())


def task10():
    """
    Task 10.

    In this function we shall test your MultiBallSimulation. Create an instance of this class using
    the default values described in the project brief and run the animation for 500 collisions.

    Watch the resulting animation carefully and make sure you aren't 
    seeing errors like balls sticking
    together or escaping the container.
    """
    mbs = MultiBallSimulation(rmax=8, nrings=3, multi=3)
    mbs.run(5000, pause_time=0.001, animate=True)


def task11():
    """
    Task 11.

    In this function we shall be quantitatively checking that the balls aren't escaping or sticking.
    To do this, create the two histograms as directed in the project script. Ensure that these two
    histogram figures are returned.

    Returns:
        tuple[Figure, Figure]: The histograms (distance from centre, inter-ball spacing).
    """
    num_collisions = 500
    mbs = MultiBallSimulation()
    distances_from_center = []
    inter_ball_distances = []
    # mbs.run(num_collisions=500, animate=True)
    for n_ in range(num_collisions):
        mbs.next_collision()
        # Calculate distances from center
        for ball in mbs.balls():
            distance = np.linalg.norm(ball.pos())
            distances_from_center.append(distance)
        # Calculate distances between each pair of balls
        for i in range(len(mbs.balls())):
            for j in range(i + 1, len(mbs.balls())):
                b_1 = mbs.balls()[i]
                b_2 = mbs.balls()[j]
                distance = np.linalg.norm(b_1.pos() - b_2.pos())
                inter_ball_distances.append(distance)
    # Plot histograms
    fig1, ax1 = plt.subplots()
    ax1.hist(distances_from_center, bins=20)
    ax1.set_title('Distances from Center')
    ax1.set_xlabel('Distance (m)')
    ax1.set_ylabel('Frequency')

    fig2, ax2 = plt.subplots()
    ax2.hist(inter_ball_distances, bins=20)
    ax2.set_title('Inter-Ball Distances')
    ax2.set_xlabel('Distance (m)')
    ax2.set_ylabel('Frequency')

    return fig1, fig2

def task12():
    """
    Task 12.

    In this function we shall check that the fundamental quantities of energy 
    and momentum are conserved. Additionally we shall investigate the 
    pressure evolution of the system. Ensure that the 4 figures outlined 
    in the project script are returned.

    Returns:
        tuple[Figure, Figure, Figure, Figure]: matplotlib Figures of the KE, 
        momentum_x, momentum_y ratios as well as pressure evolution.
    """
    mbs = MultiBallSimulation(c_radius=10., b_radius=1., b_speed=10., b_mass=1.,
                              rmax=8, nrings=3, multi=6)
    kin_e_0 = round(mbs.kinetic_energy(),5)
    mom_x_0 = mbs.momentum()[0]
    mom_y_0 = mbs.momentum()[1]
    num_collisions = 4000
    #Lists containing the ratios
    kin_e = []
    mom_x = []
    mom_y = []
    pressure = []
    time = []
    for n in range(num_collisions):
        mbs.next_collision()
        kin_e.append(mbs.kinetic_energy()/kin_e_0)
        mom_x.append(mbs.momentum()[0]/mom_x_0)
        mom_y.append(mbs.momentum()[1]/mom_y_0)
        pressure.append(mbs.pressure())
        time.append(mbs.time())


    # Plotting KE ratio over time
    fig1, ax1 = plt.subplots()
    ax1.plot(time, kin_e)
    ax1.set_title('Ratio of KE(t) / KE(0)')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Ratio')
    ax1.set_ylim(0.9,1.1)
    ax1.grid()
    # Plotting Momentum_x ratio over time
    fig2, ax2 = plt.subplots()
    ax2.plot(time, mom_x)
    ax2.set_title('Ratio of Sum Momentum_x(t) / Sum Momentum_x(0)')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Ratio')
    ax2.set_ylim(0.9,1.1)
    ax2.grid()
    # Plotting Momentum_y ratio over time
    fig3, ax3 = plt.subplots()
    ax3.plot(time, mom_y)
    ax3.set_title('Ratio of Sum Momentum_y(t) / Sum Momentum_y(0)')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Ratio')
    ax3.set_ylim(0.9,1.1)
    ax3.grid()
    # Plotting Pressure over time
    fig4, ax4 = plt.subplots()
    ax4.plot(time, pressure)
    ax4.set_title('Pressure against Time')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Pressure (Pa)')
    ax4.grid()

    return fig1, fig2, fig3, fig4

def task13():
    """
    Task 13.

    In this function we investigate how well our simulation reproduces the distributions of the IGL.
    Create the 3 figures directed by the project script, namely:
    1) PT plot
    2) PV plot
    3) PN plot
    Ensure that this function returns the three matplotlib figures.

    Returns:
        tuple[Figure, Figure, Figure]: The 3 requested figures: (PT, PV, PN)
    """
    k_b = 1.38*10**(-23)
    p_1 = []
    temp = []
    for speed in np.arange(0.1, 300, 0.1):
        mbs = MultiBallSimulation(c_radius=10., b_radius=.1, b_speed=speed, b_mass=1.,
                              rmax=8., nrings=3, multi=1)
        mbs.run(num_collisions=100)
        p_1.append(mbs.pressure())
        temp.append(mbs.t_equipartition()/k_b) #Because the method returns temperature*k_b
    p_ideal_1 = (len(mbs.balls())*k_b)/(mbs.container().volume()) * np.array(temp)
    fig1, ax1 = plt.subplots()
    ax1.scatter(temp, p_1, label='Simulation')
    ax1.plot(temp, p_ideal_1, color='g', label='Ideal')
    ax1.set_title('Pressure variation with Temperature')
    ax1.set_xlabel('Temperature (K)')
    ax1.set_ylabel('Pressure (Pa)')
    ax1.legend()
    p_2 = []
    vol = []
    for radius in np.arange(10, 20, 0.1):
        mbs = MultiBallSimulation(c_radius=radius, b_radius=1, b_speed=1, b_mass=1.,
                            rmax=8., nrings=3, multi=3)
        mbs.run(num_collisions=100, animate=False)
        p_2.append(mbs.pressure())
        vol.append(mbs.container().volume())
    p_2 = np.array(p_2)
    vol = np.array(vol)
    fig2, ax2 = plt.subplots()
    ax2.scatter(vol, p_2, label='Simulation')
    ax2.set_title('Pressure variation with volume')
    ax2.set_xlabel('Volume (m$^3$)')
    ax2.set_ylabel('Pressure (Pa)')

    ball_radius = mbs.balls()[0].radius()
    ball_volume = np.pi * ball_radius**2

    def volume_correction_factor(factor):
        """
        Calculate the sum of squared differences between the simulated pressures and the ideal gas pressures
        with a corrected volume.

        Args:
            factor (float): The volume correction factor to be applied.

        Returns:
            float: The sum of squared differences between the simulated and ideal pressures.
        """
        corrected_vol = np.array(vol) - factor * n * ball_volume
        corrected_p_ideal = len(mbs.balls()) * mbs.t_equipartition() / corrected_vol
        return np.sum((corrected_p_ideal - p_2)**2)

    n = len(mbs.balls())

    initial_guess = ball_radius
    result = minimize(volume_correction_factor, initial_guess)
    correction_factor = result.x[0]
    print(f"Correction factor: {correction_factor}")

    # Plot the corrected volume
    corrected_vol = np.array(vol) - correction_factor * n * ball_volume
    ax2.scatter(corrected_vol, p_2, label='Corrected Data')

    vol_range = np.linspace(min(corrected_vol), max(vol), 500)
    p_ideal_2 = len(mbs.balls()) * mbs.t_equipartition() / vol_range
    ax2.plot(vol_range, p_ideal_2, color='g', label='Ideal')

    ax2.legend()
    p_3 = []
    no_of_particles = []
    for mul in np.arange(1, 20, 1):
        mbs = MultiBallSimulation(c_radius=10., b_radius=0.1, b_speed=1., b_mass=1.,
                              rmax=8., nrings=3, multi=mul)
        mbs.run(num_collisions=100)
        p_3.append(mbs.pressure())
        no_of_particles.append(len(mbs.balls()))
    p_ideal_3 = np.array(no_of_particles)/(mbs.container().volume()) * mbs.t_equipartition()
    fig3, ax3 = plt.subplots()
    ax3.scatter(no_of_particles, p_3, label='Simulation')
    ax3.plot(no_of_particles, p_ideal_3, color='g', label='Ideal')
    ax3.set_title('Pressure variation with number of particles')
    ax3.set_xlabel('Number of Particles')
    ax3.set_ylabel('Pressure (Pa)')
    ax3.legend()
    return None, fig2, None

def task14():
    """
    Task 14.

    In this function we shall be looking at the divergence of our simulation from the IGL. We shall
    quantify the ball radii dependence of this divergence by plotting the temperature ratio 
    defined in the project brief.

    Returns:
        Figure: The temperature ratio figure.
    """
    radii = []
    temp_ratio = []
    for radius in np.arange(0.01,1, 0.01):
        mbs = MultiBallSimulation(b_radius=radius)
        mbs.run(num_collisions=100)
        radii.append(radius)
        k_b = 1.38*10**-23
        temp_ratio.append((mbs.t_equipartition()/k_b)/mbs.t_ideal())
    fig, axis = plt.subplots()
    axis.plot(radii, temp_ratio)
    axis.set_title('T_equipartition/T_ideal ratio with varying radii of balls')
    axis.set_xlabel('Radius (m)')
    axis.set_ylabel('Temperature ratio')

    return fig

def task15():
    """
    Task 15.

    In this function we shall plot a histogram to investigate how the speeds of the
    balls evolve from the initial value. We shall then compare this to the Maxwell-Boltzmann
    distribution. Ensure that this function returns the created histogram.

    Returns:
        Figure: The speed histogram.
    """
    k_b = 1.38*10**(-23)
    mass = 1.
    mbs = MultiBallSimulation(c_radius=15, b_speed=10, b_radius=0.1, b_mass=mass, rmax=13., nrings=8, multi=8)
    mbs.run(num_collisions=2000, animate=False)
    speed = np.array(mbs.speeds())

    fig, axis = plt.subplots()
    n_, bins, _ = axis.hist(speed, bins=50, label='Speed distribution')
    max_count = max(n_) 

    speed_sorted = np.sort(speed)
    pdf = maxwell(speed_sorted, k_b * mbs.t_ideal())
    max_pdf = max(pdf)
    # Normalize PDF so its maximum matches the histogram's maximum bin count
    normalized_pdf = pdf * (max_count / max_pdf)

    axis.plot(speed_sorted, normalized_pdf, label='Maxwell-Boltzmann PDF')
    axis.set_title('Distribution of speeds')
    axis.set_xlabel('Speed (m/s)')
    axis.set_ylabel('Frequency')
    axis.legend()
    return fig


if __name__ == "__main__":


    # Run task 9 function
    BALL_POS, BALL_VEL = task9()

    # Run task 10 function
    task10()

    # Run task 11 function
    FIG11_BALLCENTRE, FIG11_INTERBALL = task11()

    # Run task 12 function
    FIG12_KE, FIG12_MOMX, FIG12_MOMY, FIG12_PT = task12()

    # Run task 13 function
    FIG13_PT, FIG13_PV, FIG13_PN = task13()

    # Run task 14 function
    FIG14 = task14()

    # Run task 15 function
    FIG15 = task15()

    plt.show()
