# mlLearningAgents.py
# parsons/27-mar-2017
#
# A stub for a reinforcement learning agent to work with the Pacman
# piece of the Berkeley AI project:
#
# http://ai.berkeley.edu/reinforcement.html
#
# As required by the licensing agreement for the PacMan AI we have:
#
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

# This template was originally adapted to KCL by Simon Parsons, but then
# revised and updated to Py3 for the 2022 course by Dylan Cope and Lin Li

from __future__ import absolute_import
from __future__ import print_function

from collections import defaultdict
import random
import math

from pacman import Directions, GameState
from pacman_utils.game import Agent
from pacman_utils import util


class GameStateFeatures:
    """
    Wrapper class around a game state where you can extract
    useful information for your Q-learning algorithm

    WARNING: We will use this class to test your code, but the functionality
    of this class will not be tested itself
    """

    def __init__(self, state: GameState):
        """
        Args:
            state: A given game state object
        """
        self.state = state
        self.pacman = state.getPacmanPosition()
        self.ghosts = tuple(state.getGhostPositions())
        self.food = state.getFood()
        self.foodGrid = tuple(
            self.food[row][col]
            for row in range(self.food.width)
            for col in range(self.food.height)
        )

    def __hash__(self) -> int:
        """
        Returns:
            A hash value representing the state
        """
        return hash(
            (
                self.pacman,
                self.ghosts,
                self.foodGrid,
            )
        )

    def __repr__(self) -> str:
        return f"pacman: {self.pacman}, ghosts: {self.ghosts}"

    def __eq__(self, other) -> bool:
        """
        Args:
            other: Another GameStateFeatures object

        Returns:
            True if the two objects are equal, False otherwise
        """
        return (
            self.pacman,
            self.ghosts,
            self.foodGrid,
        ) == (
            other.pacman,
            other.ghosts,
            other.foodGrid,
        )


class QLearnAgent(Agent):
    def __init__(
        self,
        alpha: float = 0.2,
        epsilon: float = 0.05,
        gamma: float = 0.8,
        maxAttempts: int = 30,
        numTraining: int = 10,
    ):
        """
        These values are either passed from the command line (using -a alpha=0.5,...)
        or are set to the default values above.

        The given hyperparameters are suggestions and are not necessarily optimal
        so feel free to experiment with them.

        Args:
            alpha: learning rate
            epsilon: exploration rate
            gamma: discount factor
            maxAttempts: How many times to try each action in each state
            numTraining: number of training episodes
        """
        super().__init__()
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.gamma = float(gamma)
        self.maxAttempts = int(maxAttempts)
        self.numTraining = int(numTraining)
        # Count the number of games we have played
        self.episodesSoFar = 0
        self.qValues = defaultdict(lambda: util.Counter())
        self.frequencies = defaultdict(lambda: util.Counter())
        self.lastState = None
        self.lastAction = None

    # Accessor functions for the variable episodesSoFar controlling learning
    def incrementEpisodesSoFar(self):
        self.episodesSoFar += 1

    def getEpisodesSoFar(self):
        return self.episodesSoFar

    def getNumTraining(self):
        return self.numTraining

    # Accessor functions for parameters
    def setEpsilon(self, value: float):
        self.epsilon = value

    def getAlpha(self) -> float:
        return self.alpha

    def setAlpha(self, value: float):
        self.alpha = value

    def getGamma(self) -> float:
        return self.gamma

    def getMaxAttempts(self) -> int:
        return self.maxAttempts

    @staticmethod
    def computeReward(startState: GameState, endState: GameState) -> float:
        """
        Args:
            startState: A starting state
            endState: A resulting state

        Returns:
            The reward assigned for the given trajectory
        """
        # The reward is the difference in score between the two states
        if endState.isWin():
            return 10000
        if endState.isLose():
            return -10000
        return endState.getScore() - startState.getScore()

    def getQValue(self, state: GameStateFeatures, action: Directions) -> float:
        """
        Args:
            state: A given state
            action: Proposed action to take

        Returns:
            Q(state, action)
        """
        qValue = self.qValues[state][action]
        return qValue

    def maxQValue(self, state: GameStateFeatures) -> float:
        """
        Args:
            state: The given state

        Returns:
            q_value: the maximum estimated Q-value attainable from the state
        """
        if self.qValues[state].totalCount() == 0:
            return 0
        return max(self.qValues[state].values())

    def learn(
        self,
        state: GameStateFeatures,
        action: Directions,
        reward: float,
        nextState: GameStateFeatures,
    ):
        """
        Performs a Q-learning update

        Args:
            state: the initial state
            action: the action that was took
            nextState: the resulting state
            reward: the reward received on this trajectory
        """

        # Get the Q Value for the current state/action
        qValue = self.getQValue(state, action)

        # Compute the Q-value for the next state
        nextQValue = self.maxQValue(nextState)

        # Update the Q-value for the state-action pair
        self.qValues[state][action] = qValue + self.alpha * (
            reward + self.gamma * nextQValue - qValue
        )

    def updateCount(self, state: GameStateFeatures, action: Directions):
        """
        Updates the stored visitation counts.

        Args:
            state: Starting state
            action: Action taken
        """
        self.frequencies[state][action] += 1

    def getCount(self, state: GameStateFeatures, action: Directions) -> int:
        """
        Args:
            state: Starting state
            action: Action taken

        Returns:
            Number of times that the action has been taken in a given state
        """
        return self.frequencies[state][action]

    def explorationFn(self, utility: float, counts: int) -> float:
        """
        Computes exploration function.
        Return a value based on the counts

        HINT: Do a greed-pick or a least-pick

        Args:
            utility: expected utility for taking some action a in some given state s
            counts: counts for having taken visited

        Returns:
            The exploration value
        """

        if counts == 0:
            return float("inf")

        return 1 / counts + utility + 1

    def getAction(self, state: GameState) -> Directions:
        """
        Choose an action to take to maximise reward while
        balancing gathering data for learning

        If you wish to use epsilon-greedy exploration, implement it in this method.
        HINT: look at pacman_utils.util.flipCoin

        Args:
            state: the current state

        Returns:
            The action to take
        """
        # The data we have about the state of the game
        legal = state.getLegalPacmanActions()
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)

        # logging to help you understand the inputs, feel free to remove
        # print("Legal moves: ", legal)
        # print("Pacman position: ", state.getPacmanPosition())
        # print("Ghost positions:", state.getGhostPositions())
        # print("Food locations: ")
        # print(state.getFood())
        # print("Score: ", state.getScore())

        stateFeatures = GameStateFeatures(state)

        # Pick action with max Q value
        action = None
        if self.lastState is not None:
            lastStateFeatures = GameStateFeatures(self.lastState)
            self.updateCount(lastStateFeatures, self.lastAction)
            self.learn(
                lastStateFeatures,
                self.lastAction,
                self.computeReward(self.lastState, state),
                stateFeatures,
            )

        maxAction = None
        maxValue = 0
        for action in legal:
            expectedUtility = self.getQValue(stateFeatures, action)
            count = self.getCount(stateFeatures, action)
            exploreValue = self.explorationFn(expectedUtility, count)
            if exploreValue > maxValue:
                maxValue = exploreValue
                maxAction = action
        action = maxAction

        if action is None:
            action = random.choice(legal)

        self.lastState = state
        self.lastAction = action

        # Now pick what action to take.
        # The current code shows how to do that but just makes the choice randomly.
        return action

    def final(self, state: GameState):
        """
        Handle the end of episodes.
        This is called by the game after a win or a loss.

        Args:
            state: the final game state
        """
        print(f"Game {self.getEpisodesSoFar()} just ended!")

        lastStateFeatures = GameStateFeatures(self.lastState)
        stateFeatures = GameStateFeatures(state)

        # update Q-values
        self.updateCount(lastStateFeatures, self.lastAction)
        self.learn(
            lastStateFeatures,
            self.lastAction,
            self.computeReward(self.lastState, state),
            stateFeatures,
        )

        self.lastState = None
        self.lastAction = None

        if self.getEpisodesSoFar() == 0:
            self.setEpsilon(0.1)
            self.setAlpha(0.3)
        elif self.getEpisodesSoFar() == self.getNumTraining() / 2:
            self.setEpsilon(0.05)
            self.setAlpha(0.1)

        # Keep track of the number of games played, and set learning
        # parameters to zero when we are done with the pre-set number
        # of training episodes
        self.incrementEpisodesSoFar()
        if self.getEpisodesSoFar() == self.getNumTraining():
            msg = "Training Done (turning off epsilon and alpha)"
            print("%s\n%s" % (msg, "-" * len(msg)))
            self.setAlpha(0)
            self.setEpsilon(0)
