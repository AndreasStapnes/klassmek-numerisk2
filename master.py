import matplotlib.pyplot as plt
import numpy as np
from angularMethods import Phase as AngularPhase
from angularMethods import T,V,H
from methods import Phase as CartesianPhase

from methods import SYS





if __name__ == "__main__":
    dt = 1e-3
    release_angles = np.linspace(-20,40,60)*np.pi/180
    crash_distances = []
    for release_angle in release_angles:
        phase = AngularPhase(0.9*np.pi, 0, 0.9*np.pi, 0, dt)
        final_phase = phase.propagate_until_release(release_angle)
        system = SYS(final_phase.m2_cartesian_phase(), 0)
        system.propagate_until_crash(0, dt)
        crash_distances.append(system.crash_distance)
        print("*",end="")
    plt.figure()
    plt.plot(release_angles, crash_distances, label="dist(release_angle)")
    plt.legend()
    plt.show()

    '''phases = []
    times = []
    time = 0
    while True:
        phases.append(phase)
        times.append(time)
        phase = phase + 1
        time += phase.dt
        if phase.theta < 0 * np.pi: break

    thetas = np.array(phases).T[0]
    psis = np.array(phases).T[2]
    plt.plot(times, psis, label="psis")
    plt.plot(times, thetas, label="thetas")
    plt.legend()
    plt.show()
    plt.figure()
    plt.plot(times, [phase.m2_velocity()[0] for phase in phases])
    plt.plot(times, [phase.m2_velocity()[1] for phase in phases])
    plt.show()
    plt.figure()
    Ts = [T(phase.theta, phase.theta_prime, phase.psi, phase.psi_prime) for phase in phases]
    Vs = [V(phase.theta, phase.theta_prime, phase.psi, phase.psi_prime) for phase in phases]
    Hs = [H(phase.theta, phase.theta_prime, phase.psi, phase.psi_prime) for phase in phases]
    plt.plot(times, Ts, label="Kinetic")
    plt.plot(times, Vs, label="Potential")
    plt.plot(times, Hs, label="HAMILTONIAN")
    plt.legend()
    plt.show()'''