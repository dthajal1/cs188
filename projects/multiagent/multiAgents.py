# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util
from math import inf

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        # print(newGhostStates)
        # newFood.asList()
        # reciprocal of dist to food rather than values themselves
        # print(newFood.asList())
        
        # manhattan distance to the food
        mhDistToFoods = []
        for foodPos in newFood.asList():
            mhDistToFoods.append(manhattanDistance(foodPos, newPos))
            
        # distance to ghost
        mhDistToGhosts = []
        for ghostState in newGhostStates:
            mhDistToGhosts.append(manhattanDistance(ghostState.getPosition(), newPos))
        
        if (len(mhDistToFoods) > 0):
            # we use reciprocal because closer food => smaller denominator => higher score
            return successorGameState.getScore() + 0.5/min(mhDistToFoods)
        else:
            # closer ghost => lower score 
            return successorGameState.getScore() + min(mhDistToGhosts)

        # return successorGameState.getScore() # default

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        def minimaxAlgorithm(gameState, agentIndex, currDepth):
            if currDepth == self.depth or gameState.isWin() or gameState.isLose():
                # Terminal state => return state utility
                return self.evaluationFunction(gameState)

            if agentIndex == 0:
                # pacman's turn -- maximizer
                bestVal = -inf
                actions = gameState.getLegalActions(agentIndex)
                for action in actions:  
                    successorGameState = gameState.generateSuccessor(agentIndex, action)
                    possibleBestVal = minimaxAlgorithm(successorGameState, agentIndex + 1, currDepth)
                    bestVal = max(bestVal, possibleBestVal)
                return bestVal
            else:
                # ghosts' turn -- minimizer
                bestVal = inf
                actions = gameState.getLegalActions(agentIndex)
                for action in actions:  
                    successorGameState = gameState.generateSuccessor(agentIndex, action)
                    if agentIndex + 1 == gameState.getNumAgents():
                        # this is the last ghost so pacman will go next
                        # and since we are done with iterating through pacman and all ghosts we can now increment the depth by 1
                        possibleBestVal = minimaxAlgorithm(successorGameState, 0, currDepth + 1)
                    else:
                        # there are other ghosts left waiting for their turn
                        possibleBestVal = minimaxAlgorithm(successorGameState, agentIndex + 1, currDepth)
                    bestVal = min(bestVal, possibleBestVal)
                return bestVal

        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions(self.index)

        # we start with our pacman's turn -- maximizer
        bestVal = -inf
        actionToTake = Directions.STOP
        for action in legalMoves:
            successorGameState = gameState.generateSuccessor(self.index, action)
            possibleBestVal = minimaxAlgorithm(successorGameState, self.index + 1, 0)
            if (possibleBestVal > bestVal):
                bestVal = possibleBestVal
                actionToTake = action

        return actionToTake   

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def minimaxAlgorithm(gameState, agentIndex, currDepth, alpha, beta):
            if currDepth == self.depth or gameState.isWin() or gameState.isLose():
                # Terminal state => return state utility
                return self.evaluationFunction(gameState)

            if agentIndex == 0:
                # pacman's turn -- maximizer
                bestVal = -inf
                actions = gameState.getLegalActions(agentIndex)
                for action in actions:  
                    successorGameState = gameState.generateSuccessor(agentIndex, action)
                    possibleBestVal = minimaxAlgorithm(successorGameState, agentIndex + 1, currDepth, alpha, beta)
                    bestVal = max(bestVal, possibleBestVal)

                    # alpha-beta pruning
                    if bestVal > beta:
                        return bestVal
                    alpha = max(alpha, bestVal)
                return bestVal
            else:
                # ghosts' turn -- minimizer
                bestVal = inf
                actions = gameState.getLegalActions(agentIndex)
                for action in actions:  
                    successorGameState = gameState.generateSuccessor(agentIndex, action)
                    if agentIndex + 1 == gameState.getNumAgents():
                        # this is the last ghost so pacman will go next
                        # and since we are done with iterating through pacman and all ghosts we can now increment the depth by 1
                        possibleBestVal = minimaxAlgorithm(successorGameState, 0, currDepth + 1, alpha, beta)
                    else:
                        # there are other ghosts left waiting for their turn
                        possibleBestVal = minimaxAlgorithm(successorGameState, agentIndex + 1, currDepth, alpha, beta)
                    bestVal = min(bestVal, possibleBestVal)

                    # alpha-beta pruning
                    if bestVal < alpha:
                        return bestVal
                    beta = min(beta, bestVal)
                return bestVal

        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions(self.index)

        alpha, beta = -inf, inf # max's and min's best option on path on the root 
        # we start with our pacman's turn -- maximizer
        bestVal = -inf
        actionToTake = Directions.STOP
        for action in legalMoves:
            successorGameState = gameState.generateSuccessor(self.index, action)
            possibleBestVal = minimaxAlgorithm(successorGameState, self.index + 1, 0, alpha, beta)
            if (possibleBestVal > bestVal):
                bestVal = possibleBestVal
                actionToTake = action
            
            if bestVal > beta:
                return bestVal
            alpha = max(alpha, bestVal)

        return actionToTake  

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
