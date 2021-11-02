import matplotlib.pyplot as plt
from GalaxySimulation import AdvanceTimestep, PlotStarsFromStellarArray, PrintPercentageComplete, stellarArray, queue

nSteps = 300   # Total number of simulation timesteps

# Plot initial setup
PlotStarsFromStellarArray(stellarArray)
plt.savefig("galaxySimulationBefore.png", dpi=400)

# Advance Timestep and plot
for step in range(nSteps):
    AdvanceTimestep(step, stellarArray, queue)
    PrintPercentageComplete(step, nSteps)

PlotStarsFromStellarArray(stellarArray)
plt.savefig("galaxySimulationAfter.png", dpi=400)