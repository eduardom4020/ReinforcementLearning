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

    def actionParse(self,action):
        if action == 'East': return 0
        elif action == 'North': return 1
        elif action == 'West': return 2
        elif action == 'South': return 3
        else: return 4
    
    def initalize(self):
        self.num_features = 3 #feature0: scores; features 1: food_distance; feature 2: ghost_distance
        self.num_actions = 5    #0: right; 1: up; 2: left; 3: down; 4: stop
        self.first_action = True

    def getFeaturesCount(self):
        return self.num_features

    def getActionsCount(self):
        return self.num_actions

    def setAgentPolicy(self, state):
        self.agent_policy = state
	
    def runAgentPolicy(self, Agent):
	self.setAgentPolicy(True)
	self.agent = Agent                   

    def setAlphaAndGama(self, alpha, gama):
        self.alpha = alpha
        self.gama = gama

    def initializeWeights(self, weights):
        self.weights = weights #0: right_f1; 1: up_f1; 2: left_f1; 3: down_f1; 4: stop_f1; 5: right_f2; 6: up_f2; ...

    def getWeights(self):
        return self.weights

    def findWeight(self,action,feature_num):
        return self.actionParse(action) + self.num_actions * feature_num
	
    def getAction(self, gameState):
        if self.first_action:
            self.max_score = gameState.getNumFood() * 10.0
            
            self.corner1 = [0,0]
            self.corner2 = [gameState.getHeight()-1,gameState.getWidth()-1]
            self.maxDistance = util.manhattanDistance(self.corner1, self.corner2)
            
            self.first_action = False
        
	if self.agent_policy:
            action = self.agent.getAction(gameState)
            self.updateWeights(gameState, action)
            return action
        else:
            # Collect legal moves and successor states
            legalMoves = gameState.getLegalActions()

            # Choose one of the best actions
            Q = [self.Q(gameState, action) for action in legalMoves]
            bestQ = max(Q)
            bestIndices = [index for index in range(len(Q)) if Q[index] == bestQ]
            chosenIndex = random.choice(bestIndices)

            if self.alpha > 0:
                self.updateWeights(gameState, legalMoves[chosenIndex])

            return legalMoves[chosenIndex]  

        """get the maximum Q value for the state gameState. This is obtained
           analizing all of the LEGAL ACTIONS for that gameState.
           Note that the Q value to be analized is independent from the action
           that the agent takes
        """
        
    def Q(self, gameState, action):
        return sum( self.feature(gameState, action, feature) * self.weights[self.findWeight(action,feature)] for feature in range( self.num_features ) )
	
    def updateWeights(self, currGameState, action):
        reward = 0

        nextGameState = currGameState.generatePacmanSuccessor(action)
        legalMovesNext = nextGameState.getLegalActions()

        q_next = [self.Q(nextGameState, action_next) for action_next in legalMovesNext]
        if len(q_next) == 0:
            if nextGameState.isLose():
                reward = -500
        
            if nextGameState.isWin():
                reward = 1000
                    
            q_next = [self.Q(currGameState, action)]

        for feature_num in range( self.num_features ):        
            self.weights[self.findWeight(action,feature_num)] = self.weights[self.findWeight(action,feature_num)] + self.alpha * (reward + self.gama * max(q_next) - self.Q(currGameState, action)) * self.feature(currGameState, action, feature_num)

    def feature(self, currGameState, action, feature_num):
        if feature_num == 0:
            return currGameState.getScore() / self.max_score
        
        elif feature_num == 1:
            actualFood = currGameState.getFood()
            position = currGameState.getPacmanPosition()

            min_food_distance = 999999

            for x, food_row in enumerate(actualFood):
              for y, food in enumerate(food_row):
                if food == True:
                  food_pos = [x, y]
                  distance = util.manhattanDistance(position, food_pos)
                  if distance < min_food_distance:
                    min_food_distance = distance

            return 1 - min_food_distance / self.maxDistance

        elif feature_num == 2:
            ghostStates = currGameState.getGhostStates()
            position = currGameState.getPacmanPosition()
            
            min_ghost_distance = 999999

            for ghost in ghostStates:
                distance = util.manhattanDistance(position, ghost.getPosition())
                if distance < min_ghost_distance:
                  min_ghost_distance = distance

            return 1 - min_ghost_distance / self.maxDistance

        """elif feature_num == 3:
            walls = currGameState.getWalls()
            position = currGameState.getPacmanPosition()

            max_wall_distance = -1

            for x, wall_row in enumerate(walls):
              for y, wall in enumerate(wall_row):
                if wall == True:
                  wall_pos = [x, y]
                  distance = Space.distance(position, wall_pos)
                  if distance > max_wall_distance:
                    max_wall_distance = distance"""
                                
        #raycast features (desn't work)
        """if feature_num > 0:
            element = self.raycastPos(currGameState, action, feature_num)
            return self.elementValue(element)/self.maxElementValue()"""
    
    #OBS: in the two methods bellow, x indicate the row position (top to botton) and y the collum position(left to right)

    #shoots a ray in the direction of the action, catching targets in distance
    def raycast(self, currGameState, action, distance):
        p0 = currGameState.getPacmanPosition()
        y,x = p0
        result_set = []
        for next_pos in range(distance):
            if action == 'East':
                if y < currGameState.getWidth() - 1: y += 1
            elif action == 'North':
                if x < currGameState.getHeight() - 1: x += 1
            elif action == 'West':
                if y > 0: y -= 1 
            elif action == 'South':
                if x > 0: x -= 1
            else:
                direction = currGameState.getPacmanState().getDirection()

                if action == 'East':
                    if y < currGameState.getWidth() - 1: y += 1
                elif action == 'North':
                    if x < currGameState.getHeight() - 1: x += 1
                elif action == 'West':
                    if y > 0: y -= 1 
                elif action == 'South':
                    if x > 0: x -= 1

            result_set = [currGameState[x][y]] + result_set

        return result_set

    def raycastPos(self, currGameState, action, pos):
        result = self.raycast(currGameState, action, self.num_features)

        if pos >= len(result): 
            return 'n'
        else:
            return result[pos]

    def elementValue(self, element):
        if element == ' ': return 1                                         
        elif element == '.': return 5
        elif element == '%': return -5
        elif element == 'G': return -10
        elif element == 'o': return 20
        else: return 0

    def maxElementValue(self):
        return 20
                

class RandomAgent(game.Agent):
    
    def getAction(self, state):
        from random import randint
        
        legalMoves = state.getLegalActions()
        rand = randint(0,(len(legalMoves) - 1))
        return legalMoves[rand]  

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
  
