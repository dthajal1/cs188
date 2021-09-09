# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    # DFS uses Stack data structure (LIFO)
    fringe = util.Stack()
    fringe.push(problem.getStartState())

    # keeps track of (state, action, cost) taken to get from previous state to current state
    from collections import defaultdict
    edgeTo = defaultdict(list)

    # keeps track of nodes visted so far so we don't revisit them again
    visited = []

    # will eventually hold the list of actions taken to get from start to goal state 
    result = []

    # loop
    while not fringe.isEmpty():
        curr = fringe.pop()

        if problem.isGoalState(curr):
            for state, action, cost in edgeTo[curr]:
                result.append(action)
            return result

        if curr not in visited:
            # only want to append to visited when we explore it not when we add to the fringe
            visited.append(curr)

            for nextState, action, cost in problem.getSuccessors(curr):
                if nextState not in visited:
                    fringe.push(nextState)
                    # append previous actions taken to get to next state
                    edgeTo[nextState] = [] # we clear the edgeTo array so that we only include one recent solution (in the case there is multiple solutions)
                    for prevStates in edgeTo[curr]:
                        edgeTo[nextState].append(prevStates)
                    # append current action taken to get to next state
                    edgeTo[nextState].append((curr, action, cost))

    return result

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""    
   # BFS uses Queue data structure (FIFO)
    fringe = util.Queue()
    # if we decide to append only when expanding, it's better if we push tuple of state, action onto the queue
    fringe.push(problem.getStartState())

    # keeps track of (state, action, cost) taken to get from previous state to current state
    from collections import defaultdict
    edgeTo = defaultdict(list)

    # keeps track of nodes visted so far so we don't revisit them again
    visited = []

    # will eventually hold the list of actions taken to get from start to goal state 
    result = []

    # loop
    while not fringe.isEmpty():
        curr = fringe.pop()

        if problem.isGoalState(curr):
            for state, action, cost in edgeTo[curr]:
                result.append(action)
            return result

        if curr not in visited:
            # correct implementation only appends to visited when expanding instead of appending when adding to fringe
            visited.append(curr)

            for nextState, action, cost in problem.getSuccessors(curr):
                fringe.push(nextState)
                
                # append previous actions taken to get to next state
                edgeTo[nextState] = [] # we clear the edgeTo array so that we only include one recent solution (in the case there is multiple solutions)
                for prevStates in edgeTo[curr]:
                    edgeTo[nextState].append(prevStates)
                # append current action taken to get to next state
                edgeTo[nextState].append((curr, action, cost))

    return result


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    # UCS uses PriorityQueue data structure (FIFO based on priority -- smaller number => higher priority)
    fringe = util.PriorityQueue()
    fringe.push(problem.getStartState(), 0)

    # keeps track of (state, action, cost) taken to get from previous state to current state
    from collections import defaultdict
    edgeTo = defaultdict(list)

    # keeps track of nodes visted so far so we don't revisit them again
    visited = []

    # will eventually hold the list of actions taken to get from start to goal state 
    result = []

    # loop
    while not fringe.isEmpty():
        curr = fringe.pop()

        if problem.isGoalState(curr):
            for state, action, cost in edgeTo[curr]:
                result.append(action)
            return result

        
        if curr not in visited:
            # append to visited as we explore the nodes
            visited.append(curr)

            for nextState, action, cost in problem.getSuccessors(curr): 

                # append previous actions taken to get to next state
                # edgeTo[nextState] = [] # we clear the edgeTo array so that we only include one recent solution (in the case there is multiple solutions)
                totalCostSoFar = 0
                for prevStates in edgeTo[curr]:
                    edgeTo[nextState].append(prevStates)
                    prevState, prevAction, prevCost = prevStates
                    totalCostSoFar += prevCost

                # append current action, cost taken to get to next state
                edgeTo[nextState].append((curr, action, cost))

                # UCS picks the next cheapest node based on the total cost from start until that next node (not just the current cost from current node to next node)
                totalCostSoFar += cost
                fringe.update(nextState, totalCostSoFar)

    return result

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    # A* Search uses PriorityQueue data structure (FIFO based on priority -- smaller number => higher priority)
    fringe = util.PriorityQueue()
    fringe.push(problem.getStartState(), 0)

    # keeps track of (state, action, cost) taken to get from previous state to current state
    from collections import defaultdict
    edgeTo = defaultdict(list)

    # keeps track of nodes visted so far so we don't revisit them again
    visited = []

    # will eventually hold the list of actions taken to get from start to goal state 
    result = []

    # loop
    while not fringe.isEmpty():
        curr = fringe.pop()

        if problem.isGoalState(curr):
            for state, action, cost in edgeTo[curr]:
                result.append(action)
            return result

        
        if curr not in visited:
            # append to visited as we explore the nodes
            visited.append(curr)

            for nextState, action, cost in problem.getSuccessors(curr): 

                # append previous actions taken to get to next state
                edgeTo[nextState] = [] # we clear the edgeTo array so that we only include one recent solution (in the case there is multiple solutions)
                totalCostSoFar = 0
                for prevStates in edgeTo[curr]:
                    edgeTo[nextState].append(prevStates)
                    prevState, prevAction, prevCost = prevStates
                    totalCostSoFar += prevCost

                # append current action, cost taken to get to next state
                edgeTo[nextState].append((curr, action, cost))

                # UCS picks the next cheapest node based on the total cost from start until that next node (not just the current cost from current node to next node)
                totalCostSoFar += cost
                # A* also takes into account the heuristic value along with the total edge cost
                totalCostSoFar += heuristic(nextState, problem)
                fringe.update(nextState, totalCostSoFar)

    return result


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
