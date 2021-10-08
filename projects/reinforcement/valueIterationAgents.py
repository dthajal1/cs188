# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        # V_(k+1)(s) = max_a Q(s, a) where 
        # Q(s, a) = sum_s' T(s, a, s') * [R(s, a, s') + discount * V_k(s')]
        for _ in range(self.iterations):
            # need this because we need to read and update self.values at the same time
            newValues = util.Counter()
            # update values for all states
            for state in self.mdp.getStates():
                if self.mdp.isTerminal(state):
                    # do nothing since there are no possible legal actions from a terminal state
                    continue
                # update the new state with max q value over all possible actions
                newValues[state] = max([self.getQValue(state, action) for action in self.mdp.getPossibleActions(state)])
            self.values = newValues

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        # Q(s, a) = sum_s' T(s, a, s') * [R(s, a, s') + discount * V(s')]

        qValue = 0
        for nextState, prob in self.mdp.getTransitionStatesAndProbs(state, action):
            reward = self.mdp.getReward(state, action, nextState)
            qValue += prob * (reward + self.discount * self.values[nextState]) 

        return qValue

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        # pi(s) = argmax_a Q(s, a) where
        # Q(s, a) = sum_s' T(s, a, s') * [R(s, a, s') + discount * V(s')]

        if self.mdp.isTerminal(state):
            # do nothing because there are no possible legal actions
            return None

        qValues = util.Counter()
        for action in self.mdp.getPossibleActions(state):
            qValues[action] = self.getQValue(state, action)
        # best action is the one with highest q value
        bestPolicy = qValues.argMax()
        return bestPolicy

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        # V_k+1(s) ← max_a ∑_s′ T(s,a,s′)[R(s,a,s′)+γV_k(s′)]
        for i in range(self.iterations):
            allStates = self.mdp.getStates()
            # wrap around when all states are updated exactly once
            stateToUpdate = allStates[i % len(allStates)]
            if self.mdp.isTerminal(stateToUpdate):
                continue
            else:
                self.values[stateToUpdate] = max([self.getQValue(stateToUpdate, action) for action in self.mdp.getPossibleActions(stateToUpdate)])
                

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"

        # compute predecessors of all states
        predecessors = collections.defaultdict(set)
        for state in self.mdp.getStates():
            for action in self.mdp.getPossibleActions(state):
                for nextState, prob in self.mdp.getTransitionStatesAndProbs(state, action):
                    if state not in predecessors[nextState]:
                        predecessors[nextState].add(state)

        # Initialize an empty priority queue
        pq = util.PriorityQueue()
        # push the error (actual - whatWeGot) on to the pq
        for state in self.mdp.getStates():
            if not self.mdp.isTerminal(state):
                bestQValue = max([self.getQValue(state, action) for action in self.mdp.getPossibleActions(state)])
                diff = abs(self.values[state] - bestQValue)

                # we push negative value because pq is a min heap but we want
                # to prioritize updating states that have higher error
                pq.update(state, -diff)

        # algo
        for _ in range(self.iterations):
            if pq.isEmpty():
                break  
            highestErrState = pq.pop()
            if not self.mdp.isTerminal(highestErrState):
                self.values[highestErrState] = max([self.getQValue(highestErrState, action) for action in self.mdp.getPossibleActions(highestErrState)])
            
            for predecessor in predecessors[highestErrState]:
                if not self.mdp.isTerminal(predecessor):
                    bestQValue = max([self.getQValue(predecessor, action) for action in self.mdp.getPossibleActions(predecessor)])
                    diff = abs(self.values[predecessor] - bestQValue)
                    if diff > self.theta:
                        pq.update(predecessor, -diff)
