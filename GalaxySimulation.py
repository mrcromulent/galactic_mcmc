import numpy as np
np.random.seed(2)
import matplotlib.pyplot as plt

## Set simulation parameters
Nr          = 149   # Number of rings
Psw         = 0.183 # Probability of new star formation
Vr          = 2     # Rotational speed of galaxy, Cells/timestep
Vsw         = 0.2   # Supernova shockwave velocity, Cells/timestep
innerRing   = 120   # Rings in which stars can be initialised
initProb    = 0.02  # Number of stars initially spawned as a percentage of possible cells

def NumberOfCellsInNRings(n):
    '''
    Returns the number of cells in n polar rings. There are 6n cells in the nth ring so this is equal to 6 x the nth triangular number
    '''
    return int(6 * (n ** 2 + n)/2)

## Auxillary quantities
#offset     = (np.pi/6)  # angular offset in radians
delaySteps  = int(1/Vsw) # Timesteps before new star formed
nCells      = NumberOfCellsInNRings(Nr)
nCellsInner = NumberOfCellsInNRings(innerRing)


# Preinitialise Stellar Array
# The stellarArray is a long vector with nCells entries. A 1 corresponds to a star. The linear index can be translated into (ring,index) coordinates
stellarArray                = np.zeros(nCells)
stellarLocations            = np.random.choice([0, 1], size=nCellsInner, p=[1-initProb, initProb])
stellarArray[0:nCellsInner] = stellarLocations

# Initialise the queue
# The queue holds the locations of new stars to be added to the stellarArray
queue = [[],[],[],[],[]]

def PrintPercentageComplete(step, nSteps):
    '''
    Print a completion percentage during the simulation, in steps of 10%
    '''
    if (step % round(nSteps/10) == 0):
         print(str(step/nSteps* 100)  + " pc compete") 

def FindNeighbours(ring, idx, nRings):
    '''
    FindNeighbours returns the six 'neighbour' cells to (ring,idx) which the supernova shockwave will be propagated to
    '''
    
    sector              = np.ceil(idx/ring)
    numNeighbours       = 0
    neighboursFound     = []

    # neighbours on same ring
    numNeighbours   += 2
    idxM1, _, idxP1 = AdjIndices(ring, idx)
    neighboursFound.append([ring, idxM1])
    neighboursFound.append([ring, idxP1])

    # neighbours on outer ring
    if ring < nRings:
        numNeighbours += 2
        idxM1, idxP0, idxP1 = AdjIndices(ring + 1, idx + sector)
        neighboursFound.append([ring + 1, idxP0])
        neighboursFound.append([ring + 1, idxM1])

    # neighbours on inner ring
    if ring > 1:
        numNeighbours += 2
        idxM1, idxP0, idxP1 = AdjIndices(ring - 1, idx - sector)
        neighboursFound.append([ring - 1, idxP0])
        neighboursFound.append([ring - 1, idxP1])

    return (neighboursFound, numNeighbours)

def AdjIndices(ring, idx):
    '''
    AdjIndices returns the circular the adjacent indices to idx on ring, wrapping maxIdx + 1 to 1
    '''

    maxIdx = 6 * ring

    idxP0 = idx
    while idxP0 <= 0:
        idxP0 += maxIdx 

    # Find the below index
    trialIdx = idxP0 - 1
    if trialIdx == 0:
        idxM1 = maxIdx
    else:
        idxM1 = trialIdx

    # Find the above index
    trialIdx = idxP0 + 1
    if trialIdx > maxIdx:
        idxP1 = 1
    else:
        idxP1 = trialIdx

    return (int(idxM1), int(idxP0), int(idxP1))

def RingIdxFromLinearIdx(linIdx):
    '''
    This function takes the linearIdx (the location in stellarArray) and returns the (ringIdx, circIdx) equivalent
    '''
    
    ringIdx = 1
    circIdx = linIdx + 1

    while circIdx > 6 * ringIdx:
        circIdx -= 6 * ringIdx
        ringIdx += 1

    return (ringIdx, circIdx)

def RThetaFromRingIdx(ring, idx):
    '''
    This function converts an (ring, idx) coordinate pair to polar coordinates (r,theta) for plotting 
    '''
    offset      = (np.pi/6)/ring  # angular offset in radians
    numElems    = 6 * ring
    theta       = offset + (idx - 1)/(numElems) * 2 * np.pi

    return (ring, theta)

def PlotStarsFromStellarArray(stellarArray):
    '''
    This function produces a matplotlib.pyplot.polar plot of the stars in stellarArray
    '''
    radii   = []
    theta   = []
    labels  = []

    # For each star in stellarArray, find its corresponding polar coordinate pair and plot them
    for index, val in np.ndenumerate(stellarArray):
        if val == 1:

            ring,idx = RingIdxFromLinearIdx(index[0])
            r,th = RThetaFromRingIdx(ring, idx)
            lab = str((ring, idx))

            radii.append(r)
            theta.append(th)
            labels.append(lab)

    # Plot 
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='polar')
    ax.scatter(theta, radii, marker='o', alpha=0.4, s=2)
    ax.set_xticks(np.pi/180. * np.linspace(0,  360, 6, endpoint=False))
    ax.set_thetalim(0, 2 * np.pi)
    ax.set_rticks([])
    # plt.show()

def RotateStellarArrayCells(stellarArray):
    '''
    RotateStellaArrayCells compensates for the rotation rate of the galaxy Vr by rotating each ring in stellarArray to its new location
    '''

    for ring in range(1,Nr+1):
        marker = 6 * (ring - 1)

        currRing = stellarArray[marker:marker+6]
        rotdRing = np.roll(currRing, Vr)

        stellarArray[marker:marker+6] = rotdRing

    return stellarArray



def AdvanceTimestep(stepNumber, stellarArray, queue):
    if (stepNumber % delaySteps == 0):
        for index, val in np.ndenumerate(stellarArray):
            if val == 1:

                # For each star in stellarArray ...
                linearIdx = index[0]

                # Make the star go nova
                stellarArray[linearIdx] = 0

                # Find its position in ring/idx coordinates
                ring,idx = RingIdxFromLinearIdx(linearIdx)

                # Find its neighbours
                nghbrs, nNghbrs = FindNeighbours(ring, idx, Nr)

                # Determine if any new stars will be formed
                for i in range(nNghbrs):
                    newStar = np.random.choice([True,False],p=[Psw, 1-Psw])

                    if newStar:
                        newStarLoc = nghbrs[i]
                        AddStarToQueue(newStarLoc, queue)

        # Rotate all the cells
        stellarArray = RotateStellarArrayCells(stellarArray)

    # Update the queue, adding new stars and adding a new empty queue item at the end
    UpdateQueue(stellarArray, queue)

def AddStarToQueue(newStarLoc, queue):
    '''
    When a new star is to be formed after delaySteps, this function adds it to the queue
    '''
    queue[delaySteps - 1].append(newStarLoc)

def UpdateQueue(stellarArray, queue):
    '''
    UpdateQueue adds stars in the 0th element of the queue to stellarArray. It then pops this element and adds a new empty element at the end
    '''

    # Grab the 0th element
    currQueue = queue[0]

    # Put scheduled stars into stellarArray
    for i in range(len(currQueue)):
        ring = currQueue[i][0]

        oldIdx = currQueue[i][1]
        newIdx = (oldIdx + delaySteps * Vr) % (6 * ring)

        if newIdx == 0:
            newIdx = 6 * ring

        linearIdx = LinearIdxFromRingIdx(ring, newIdx)
        stellarArray[linearIdx] = 1

    # Remove the just-added stars from the queue
    queue.pop(0)
    queue.append([])
    

def LinearIdxFromRingIdx(ring, circIdx):
    '''
    This function translates from (ring, circIdx) representation to a linearIdx, its location in stellarArray
    '''

    return NumberOfCellsInNRings(ring-1) + circIdx - 1
