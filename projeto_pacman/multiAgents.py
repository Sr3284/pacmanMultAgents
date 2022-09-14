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

from game import Agent

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    #return currentGameState.getScore()
    return betterEvaluationFunction(currentGameState)

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
        Returns the child game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        bestValue, a = self.minimax(gameState, 0, 0)
        act = gameState.getLegalActions(0)
        return act[a]


    def minimax(self, gameState, depth, agentIndex):
        if gameState.isLose() or gameState.isWin() or depth == self.depth:
            return scoreEvaluationFunction(gameState), None
        
        numAgents = gameState.getNumAgents()
        if numAgents-1 == agentIndex:
            depth = depth+1

        agentIndex = agentIndex % gameState.getNumAgents()

        if agentIndex == 0:
            bestValue = float("-inf")
            melhorAcao = -1
            actions = gameState.getLegalActions(agentIndex)
            for a, action in enumerate(actions):
                filho = gameState.generateSuccessor(0, action)
                val, _= self.minimax(filho, depth, agentIndex+1)
                if val > bestValue:
                    bestValue = val
                    melhorAcao = a
            return bestValue, melhorAcao
                

        else:
            bestValue = float("inf")
            actions = gameState.getLegalActions(agentIndex)
            for a, action in enumerate(actions):
                filho = gameState.generateSuccessor(agentIndex, action)
                val, _= self.minimax(filho, depth, agentIndex+1)
                if val < bestValue:
                    bestValue = val
            return bestValue, None

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """ 
        "*** YOUR CODE HERE ***"
        bestValue, a = self.alphabeta(gameState, 0, 0, float("-inf"), float("inf"))
        act = gameState.getLegalActions(0)
        return act[a]

    def alphabeta(self, gameState, depth, agentIndex, b, c):
        if gameState.isLose() or gameState.isWin() or depth == self.depth:
            return scoreEvaluationFunction(gameState), None
        
        numAgents = gameState.getNumAgents()
        if numAgents-1 == agentIndex:
            depth = depth+1

        agentIndex = agentIndex % gameState.getNumAgents()

        if agentIndex == 0:
            bestValue = float("-inf")
            melhorAcao = -1
            actions = gameState.getLegalActions(agentIndex)
            for a, action in enumerate(actions):
                filho = gameState.generateSuccessor(0, action)
                val, _= self.alphabeta(filho, depth, agentIndex+1, b, c)
                if val > bestValue:
                    bestValue = val
                    melhorAcao = a

                if val > b:
                    b = val
                if c < b:
                    break

            return bestValue, melhorAcao
                

        else:
            bestValue = float("inf")
            actions = gameState.getLegalActions(agentIndex)
            for a, action in enumerate(actions):
                filho = gameState.generateSuccessor(agentIndex, action)
                val, _= self.alphabeta(filho, depth, agentIndex+1, b, c)
                if val < bestValue:
                    bestValue = val

                if val < c:
                    c = val
                if c < b:
                    break

            return bestValue, None

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
        bestValue, a = self.expectimax(gameState, 0, 0)
        act = gameState.getLegalActions(0)
        return act[a]


    def expectimax(self, gameState, depth, agentIndex):
        if gameState.isLose() or gameState.isWin() or depth == self.depth:
            return scoreEvaluationFunction(gameState), None
        
        numAgents = gameState.getNumAgents()
        if numAgents-1 == agentIndex:
            depth = depth+1

        agentIndex = agentIndex % gameState.getNumAgents()

        if agentIndex == 0: 
            bestValue = float("-inf")
            melhorAcao = -1
            actions = gameState.getLegalActions(agentIndex)
            for a, action in enumerate(actions):
                filho = gameState.generateSuccessor(0, action)
                val, _= self.expectimax(filho, depth, agentIndex+1)
                if val > bestValue:
                    bestValue = val
                    melhorAcao = a
            return bestValue, melhorAcao
                
        
        else: 
            bestValue = float("inf")
            media = 0
            val = 0
            actions = gameState.getLegalActions(agentIndex)
            qtd = len(actions)
            for a, action in enumerate(actions):
                filho = gameState.generateSuccessor(agentIndex, action)
                val, _= self.expectimax(filho, depth, agentIndex+1)
                media = media + val
            return media/qtd, None


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: Levamos em consideração a comida mais próxima do pacman e a posição dos fantasmas, os detalhes estão comentados por blocos ao longo do código da função.
    """
    pacmanPos = currentGameState.getPacmanPosition()
    ghosts = currentGameState.getGhostStates()

    ghostsHunting = []
    ghostsFleeing = []

    HuntPos = []
    FleePos = []

    # Separa os fantasmas em duas listas: hunt -> está invunerável, flee -> está vunerável
    for ghost in ghosts:
        if ghost.scaredTimer:
            ghostsFleeing.append(ghost)
        else:
            ghostsHunting.append(ghost)

    # Pega as distâncias dos fantasmas, cápsulas e comidas
    for ghost in ghostsHunting:
        HuntPos.append(manhattanDistance(pacmanPos, ghost.getPosition()))

    for ghost in ghostsFleeing:
        FleePos.append(manhattanDistance(pacmanPos, ghost.getPosition()))

    food = currentGameState.getFood().asList()
    nFood = len(food)
    nCapsules = len(currentGameState.getCapsules())
    closestFood = 1

    score = currentGameState.getScore()

    foodDis = [manhattanDistance(pacmanPos, foodPos) for foodPos in food]

    # Pega a comida mais perto do pacman
    if nFood > 0:
        closestFood = min(foodDis)

    # Se tiver um fantasma invunerável muito perto o pacman passa a priorizar a fuga
    for position in HuntPos:
        if position < 2:
            closestFood = 9999

    # Se tiver algum fantasma vunerável no mapa, ele passa a ser a prioridade do pacman
    for position in FleePos:
        if position < 9999:
            closestFood = position

    # Combinação das métricas
    return ((10 * (1.0 / closestFood)) + (200 * score) + (-100 * nFood) + (-10 * nCapsules))

# Abbreviation
better = betterEvaluationFunction

