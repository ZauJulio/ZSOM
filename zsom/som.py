import numpy as np
import pandas as pd

def coorToHex(x, y):
    """Convert Cartesian coordinates to hexagonal tiling coordinates.

        Args:
            x (float): position along the x-axis of Cartesian coordinates.
            y (float): position along the y-axis of Cartesian coordinates.
            
        Returns:
            array: a 2d array containing the coordinates in the new space.
    """

    newy = y*2/np.sqrt(3)*3/4
    newx = x
    if y%2:
        newx += 0.5
    return [newx, newy]

class SOM:
    """ Kohonen SOM Network class. """

    def __init__(self, netHeight, netWidth, data, startLearnRate, PBC=True, epochs=-1, verbose=True):
        """Initialise the SOM network.
        Args:
            netHeight (int): Number of nodes along the first dimension.
            netWidth (int): Numer of nodes along the second dimension.
            data (np.array or list): N-dimensional dataset.
            show (tuple or list):
                [0] During training : bool
                [1] In the end : bool
        """
        self.PBC = PBC
        self.nodeList = []
        self.data = data

        self.epochs = epochs
        self.verbose = verbose
        self.netHeight = netHeight
        self.netWidth = netWidth

        self.__start__(startLearnRate)
        self.__initNodes__(self.__initMinMax__())
        self.__train__()
        self.ACT()

    def __initMinMax__(self):
        min, max = [], []
        for i in range(self.data.values.shape[1]):
            min.append(np.min(self.data.values[:, i]))
            max.append(np.max(self.data.values[:, i]))
        return (min, max)

    def __initNodes__(self, minMax):
        for x in range(self.netWidth):
            for y in range(self.netHeight):
                self.nodeList.append(Node(x, y, self.data.values.shape[1],
                                          self.netHeight,
                                          self.netWidth,
                                          PBC=self.PBC,
                                          minVal=minMax[0],
                                          maxVal=minMax[1]))

    def __initVector__(self):
        """ Init input of update with bootstrap-like method """
        self.inputVec = []
        for i in range(self.epochs):
            self.inputVec.append(self.data.values[np.random.randint(0, self.data.values.shape[0]), :].reshape(np.array([self.data.values.shape[1]])))

    def __start__(self, startLearnRate):
        if (self.epochs == -1):
            self.epochs = self.data.values.shape[0]*10
        else:
            self.epochs = self.epochs

        self.startSigma = max(self.netHeight, self.netWidth)/2
        self.startLearnRate = startLearnRate
        self.tau = self.epochs/np.log(self.startSigma)
        self.__initVector__()

    def __train__(self):
        self.__initVector__()
        # self.inputVec = self.data.values
        for i in range(self.epochs):
            self.__verbose__(i)
            bmu = self.find_bmu(self.inputVec[i])
            for node in self.nodeList:
                self.updateSigma(i)
                self.updateLrate(i)
                node.update_weights(self.inputVec[i],
                                    self.sigma,
                                    self.lrate,
                                    bmu)

    def __verbose__(self, i):
        if self.verbose:
            print(("\rTraining: %.1f" %
                   ((i/self.epochs)*100.0)+"%"), end=' ')
            if i == self.epochs:
                print("\rTraining SOM... done!")

    def updateSigma(self, i):
        self.sigma = self.startSigma * np.exp(-i/self.tau)

    def updateLrate(self, i):
        self.lrate = self.startLearnRate * np.exp(-i/self.epochs)

    def find_bmu(self, vec):
        """  """
        minVal = np.finfo(np.float).max
        for node in self.nodeList:
            dist = node.get_distance(vec)
            if dist < minVal:
                minVal = dist
                bmu = node
        return bmu

    def ACT(self):
        """ [ [x,y], [weights], [data] ] """
        self.act = []
        for x in range(self.netHeight):
            for y in range(self.netWidth):
                cx, cy = coorToHex(x, y)
                cy = round(cy, 8)
                self.act.append([[cx, cy], [], []])

        i = 0
        for day in self.data.index:
            neuron = self.find_bmu(self.data.loc[day].values)
            posNeuron = [neuron.pos[0], round(neuron.pos[1], 8)]
            k = sum([i if posNeuron == self.act[i][0] else 0 for i in range(len(self.act))])
            self.act[k][1] = neuron.weights
            self.act[k][2].append(self.data.loc[day])
            i+=1


class Node:
    """ Single Kohonen SOM Node class. """

    def __init__(self, x, y, numWeights, netHeight, netWidth, PBC, minVal=[], maxVal=[], pcaVec=[], weiArray=[]):
        """Initialise the SOM node.

        Args:
            x (int): Position along the first network dimension.
            y (int): Position along the second network dimension
            numWeights (int): Length of the weights vector.
            netHeight (int): Network height, needed for periodic boundary conditions (PBC)
            netWidth (int): Network width, needed for periodic boundary conditions (PBC)
            PBC (bool): Activate/deactivate periodic boundary conditions.
            minVal(np.array, optional): minimum values for the weights found in the data
            maxVal(np.array, optional): maximum values for the weights found in the data
            pcaVec(np.array, optional): Array containing the two PCA vectors.
            weiArray (np.array, optional): Array containing the weights to give
                to the node if a file was loaded.

        """

        self.PBC = PBC
        self.pos = coorToHex(x, y)
        self.weights = []

        self.netHeight = netHeight
        self.netWidth = netWidth

        if weiArray == [] and pcaVec == []:
            # select randomly in the space spanned by the data
            for i in range(numWeights):
                self.weights.append(np.random.random() *
                                    (maxVal[i]-minVal[i])+minVal[i])
        elif weiArray == [] and pcaVec != []:
            # select uniformly in the space spanned by the PCA vectors
            self.weights = (x-self.netWidth/2)*2.0/self.netWidth * \
                pcaVec[0] + (y-self.netHeight/2)*2.0/self.netHeight * pcaVec[1]
        else:
            for i in range(numWeights):
                self.weights.append(weiArray[i])

    def get_distance(self, vec):
        """Calculate the distance between the weights vector of the node and a given vector.

        Args:
            vec (np.array): The vector from which the distance is calculated.

        Returns: 
            (float): The distance between the two weight vectors.
        """

        sum = 0
        if len(self.weights) == len(vec):
            for i in range(len(vec)):
                sum += (self.weights[i]-vec[i])**2
            return np.sqrt(sum)
        else:
            raise ValueError("Error: dimension of nodes != input data dimension!")

    def get_nodeDistance(self, node):
        """Calculate the distance within the network between the node and another node.

        Args:
            node (somNode): The node from which the distance is calculated.

        Returns:
            (float): The distance between the two nodes.

        """

        if self.PBC == True:

            """ Hexagonal Periodic Boundary Conditions """

            if self.netHeight % 2 == 0:
                offset = 0
            else:
                offset = 0.5

            return np.min([np.sqrt((self.pos[0]-node.pos[0])*(self.pos[0]-node.pos[0])
                                   + (self.pos[1]-node.pos[1])*(self.pos[1]-node.pos[1])),
                           # right
                           np.sqrt((self.pos[0]-node.pos[0]+self.netWidth)*(self.pos[0]-node.pos[0]+self.netWidth)\
                                   + (self.pos[1]-node.pos[1])*(self.pos[1]-node.pos[1])),
                           # bottom
                           np.sqrt((self.pos[0]-node.pos[0]+offset)*(self.pos[0]-node.pos[0]+offset)\
                                   + (self.pos[1]-node.pos[1]+self.netHeight*2/np.sqrt(3)*3/4)*(self.pos[1]-node.pos[1]+self.netHeight*2/np.sqrt(3)*3/4)),
                           # left
                           np.sqrt((self.pos[0]-node.pos[0]-self.netWidth)*(self.pos[0]-node.pos[0]-self.netWidth)\
                                   + (self.pos[1]-node.pos[1])*(self.pos[1]-node.pos[1])),
                           # top
                           np.sqrt((self.pos[0]-node.pos[0]-offset)*(self.pos[0]-node.pos[0]-offset)\
                                   + (self.pos[1]-node.pos[1]-self.netHeight*2/np.sqrt(3)*3/4)*(self.pos[1]-node.pos[1]-self.netHeight*2/np.sqrt(3)*3/4)),
                           # bottom right
                           np.sqrt((self.pos[0]-node.pos[0]+self.netWidth+offset)*(self.pos[0]-node.pos[0]+self.netWidth+offset)\
                                   + (self.pos[1]-node.pos[1]+self.netHeight*2/np.sqrt(3)*3/4)*(self.pos[1]-node.pos[1]+self.netHeight*2/np.sqrt(3)*3/4)),
                           # bottom left
                           np.sqrt((self.pos[0]-node.pos[0]-self.netWidth+offset)*(self.pos[0]-node.pos[0]-self.netWidth+offset)\
                                   + (self.pos[1]-node.pos[1]+self.netHeight*2/np.sqrt(3)*3/4)*(self.pos[1]-node.pos[1]+self.netHeight*2/np.sqrt(3)*3/4)),
                           # top right
                           np.sqrt((self.pos[0]-node.pos[0]+self.netWidth-offset)*(self.pos[0]-node.pos[0]+self.netWidth-offset)\
                                   + (self.pos[1]-node.pos[1]-self.netHeight*2/np.sqrt(3)*3/4)*(self.pos[1]-node.pos[1]-self.netHeight*2/np.sqrt(3)*3/4)),
                           # top left
                           np.sqrt((self.pos[0]-node.pos[0]-self.netWidth-offset)*(self.pos[0]-node.pos[0]-self.netWidth-offset)\
                                   + (self.pos[1]-node.pos[1]-self.netHeight*2/np.sqrt(3)*3/4)*(self.pos[1]-node.pos[1]-self.netHeight*2/np.sqrt(3)*3/4))])

        else:
            return np.sqrt((self.pos[0]-node.pos[0])*(self.pos[0]-node.pos[0])
                           + (self.pos[1]-node.pos[1])*(self.pos[1]-node.pos[1]))

    def update_weights(self, inputVec, sigma, lrate, bmu):
        dist = self.get_nodeDistance(bmu)
        dist = dist  # * 0.5
        gauss = np.exp(-dist*dist/(2*sigma*sigma))

        for i in range(len(self.weights)):
            self.weights[i] = self.weights[i] - gauss * \
                lrate*(self.weights[i]-inputVec[i])
