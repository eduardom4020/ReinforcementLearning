# pacmanAgents.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from pacman import Directions
from game import Agent
from multiAgents import ReflexAgent
import random
import game
import util

#OBS: This is a OFF-POLICY QLarning agent, so it needs to run a training set before play a game
class QLearningAgent(Agent):
        #change the value of self.num_features to specify how many feature functions are used, and define the feature in the array
        def initalizeFeatures(self):
                self.num_features = 1

        def getFeaturesCount(self):
                return self.num_features

        def setAgentPolicy(self, state):
                self.agent_policy = state
	
	def runAgentPolicy(self, Agent):
		self.setAgentPolicy(True)
		self.agent = Agent
		
        def initializeWeights(self, weights):
                self.weights = []
                for w in weights:
                        self.weights.append(w)                        

        def setAlphaAndGama(self, alpha, gama):
                self.alpha = alpha
                self.gama = gama

        def getWeights(self):
                return self.weights                
	
	def getAction(self, gameState):
		if self.agent_policy == True:
                        action = self.agent.getAction(gameState)
                        self.updateWeights(gameState)
                        return action
                else:
                        # Collect legal moves and successor states
                        legalMoves = gameState.getLegalActions()

                        # Choose one of the best actions
                        evaluations = [self.Q(gameState, action) for action in legalMoves]
                        bestEvaluation = max(evaluations)
                        bestIndices = [index for index in range(len(evaluations)) if evaluations[index] == bestEvaluation]
                        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

                        "Add more of your code here if you want to"

                        return legalMoves[chosenIndex]  

        """get the maximum Q value for the state gameState. This is obtained
           analizing all of the LEGAL ACTIONS for that gameState.
           Note that the Q value to be analized is independent from the action
           that the agent takes
        """
        
        def Q(self, gameState, action):
                legalMoves = gameState.getLegalActions()
                evaluations = [self.featuresSum(gameState, action) for action in legalMoves]
                #print evaluations
                if(len(evaluations) > 0):
                        bestEvaluation = max(evaluations)
                        bestIndices = [index for index in range(len(evaluations)) if evaluations[index] == bestEvaluation]
                        maxQ = random.choice(bestIndices)
                else:
                        maxQ = 0
                        
                return maxQ
	
	def updateWeights(self, currGameState):
                reward = 0

                legalMoves = currGameState.getLegalActions()
                nextGameStates = [currGameState.generatePacmanSuccessor(action) for action in legalMoves]
                #print nextGameState

                if(currGameState.isLose == True):
                        reward = -500
                
                if(currGameState.isWin == True):
                        reward = 1000

                for i, w in enumerate( self.weights ):
                        partial_w = [w + self.alpha * ((reward + self.gama * self.Q(nextGameState, action)) - self.Q(currGameState, action)) * self.feature(currGameState, action, i) for nextGameState in nextGameStates]
                        self.weights[i] = max( partial_w )
        
	def featuresSum(self, currGameState, action):        
                return sum( self.feature(currGameState, action, feature) for feature in range( self.num_features ) )

        def feature(self, currGameState, action, feature_num):
                #teacher's sugestion: use only scores value for evaluation. It can permits that the agent adapt itself to many others levels
                nextGameState = currGameState.generatePacmanSuccessor(action)
                
                if feature_num == 0:
                        return nextGameState.getScore() * (1/self.weights[feature_num])
	

class LeftTurnAgent(game.Agent):
  "An agent that turns left at every opportunity"
  
  def getAction(self, state):
    legal = state.getLegalPacmanActions()
    current = state.getPacmanState().configuration.direction
    if current == Directions.STOP: current = Directions.NORTH
    left = Directions.LEFT[current]
    if left in legal: return left
    if current in legal: return current
    if Directions.RIGHT[current] in legal: return Directions.RIGHT[current]
    if Directions.LEFT[left] in legal: return Directions.LEFT[left]
    return Directions.STOP

class GreedyAgent(Agent):
  def __init__(self, evalFn="scoreEvaluation"):
    self.evaluationFunction = util.lookup(evalFn, globals())
    assert self.evaluationFunction != None
        
  def getAction(self, state):
    # Generate candidate actions
    legal = state.getLegalPacmanActions()
    if Directions.STOP in legal: legal.remove(Directions.STOP)
      
    successors = [(state.generateSuccessor(0, action), action) for action in legal] 
    scored = [(self.evaluationFunction(state), action) for state, action in successors]
    bestScore = max(scored)[0]
    bestActions = [pair[1] for pair in scored if pair[0] == bestScore]
    return random.choice(bestActions)
  
def scoreEvaluation(state):
  return state.getScore()  
  
