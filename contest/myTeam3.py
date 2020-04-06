# myTeam3.py
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


from captureAgents import CaptureAgent
import random, time, util
from game import Directions
import game
from util import nearestPoint

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'Attacker', second = 'Defender'):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """

    # The following line is an example only; feel free to change it.
    return [eval(first)(firstIndex), eval(second)(secondIndex)]

class attackerAction():
    def __init__(self, agent, index, gameState):
        self.agent = agent
        self.index = index

        # Get the boundary
        if self.agent.red:
            boundary = (gameState.data.layout.width - 2) / 2
        else:
            boundary = ((gameState.data.layout.width - 2) / 2) + 1
        self.boundary = []
        for i in range(1, gameState.data.layout.height - 1):
            if not gameState.hasWall(boundary, i):
                self.boundary.append((boundary, i))
        self.targetFood = None
        self.trapTimer = 0
        self.preAction = None

    def getSuccessor(self, gameState, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

    def getTargetFood(self, gameState):
        position = gameState.getAgentState(self.index).getPosition()
        foodList = self.agent.getFood(gameState).asList()
        if self.targetFood is None or self.targetFood not in foodList:
            minDistToFood = 1000
            if len(foodList) > 0:
                for food in foodList:
                    foodDistance = self.agent.getMazeDistance(position, food)
                    if foodDistance < minDistToFood:
                        minDistToFood = foodDistance
                        self.targetFood = food
        else:
            if self.trapTimer >= 5:
                minDistToFood = 1000
                preTarget = self.targetFood
                if len(foodList) > 0:
                    for food in foodList:
                        if food != preTarget:
                            foodDistance = self.agent.getMazeDistance(position, food)
                            if foodDistance < minDistToFood:
                                minDistToFood = foodDistance
                                self.targetFood = food
                    self.trapTimer = 0
        return self.targetFood


    def getFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        position = successor.getAgentState(self.index).getPosition()

        # Use the distance to the target food as a feature
        target = self.getTargetFood(gameState)
        features['distToFood'] = self.agent.getMazeDistance(position, target)

        # Use the distance to the nearest capsule as a feature
        minDistToCapsule = 1000
        capsulesList = self.agent.getCapsules(gameState)
        if len(capsulesList) > 0:
            for capsule in capsulesList:
                capsuleDistance = self.agent.getMazeDistance(position, capsule)
                if capsuleDistance < minDistToCapsule:
                    minDistToCapsule = capsuleDistance
            features['distToCapsule'] = minDistToCapsule
        else:
            features['distToCapsule'] = 0

        enemies = []
        for i in self.agent.getOpponents(gameState):
            enemies.append(successor.getAgentState(i))
        # Use the distance to nearest visible ghost as a feature
        visibleGhost = []
        for enemy in enemies:
            if enemy.getPosition() != None and not enemy.isPacman:
                visibleGhost.append(enemy)
        if len(visibleGhost) > 0:
            minDistToGhost = 6
            for ghost in visibleGhost:
                enemyPos = ghost.getPosition()
                enemyDistance = self.agent.getMazeDistance(position, enemyPos)
                if enemyDistance < minDistToGhost:
                    minDistToGhost = enemyDistance
            features['distToGhost'] = minDistToGhost
        else:
            # If no ghost is visible, ignore the actual distance
            features['distToGhost'] = 0
        # Use the distance to nearest visible Pacman as a feature
        visiblePacman = []
        for enemy in enemies:
            if enemy.getPosition() != None and enemy.isPacman:
                visiblePacman.append(enemy)
        if len(visiblePacman) > 0:
            minDistToPacman = 6
            for pacman in visiblePacman:
                enemyPos = pacman.getPosition()
                enemyDistance = self.agent.getMazeDistance(position, enemyPos)
                if enemyDistance < minDistToPacman:
                    minDistToPacman = enemyDistance
            features['distToPacman'] = minDistToPacman
        else:
            # If no Pacman is visible, ignore the actual distance
            features['distToPacman'] = 0

        # Use the number of carried food as a feature
        features['foodCarrying'] = successor.getAgentState(self.index).numCarrying

        # Use the distance to home as a feature
        minDistToHome = 1000
        for bound in self.boundary:
            homeDistance = self.agent.getMazeDistance(position, bound)
            if homeDistance < minDistToHome:
                minDistToHome = homeDistance
        if successor.getAgentState(self.index).isPacman:
            features['distToHome'] = minDistToHome
        else:
            features['distToHome'] = 0

        # Use the score as a feature
        features['score'] = self.agent.getScore(successor)

        return features

    def getWeights(self, gameState, action):
        weights = util.Counter()
        successor = self.getSuccessor(gameState, action)

        enemies = []
        for i in self.agent.getOpponents(gameState):
            enemies.append(successor.getAgentState(i))
        visibleGhost = []
        visiblePacman = []
        for enemy in enemies:
            if not enemy.isPacman and enemy.getPosition() != 0:
                visibleGhost.append(enemy)
            if enemy.isPacman and enemy.getPosition() != 0:
                visiblePacman.append(enemy)
        weights['score'] = 100

        # If the attack is a ghost, try to kill nearby pacman
        if not successor.getAgentState(self.index).isPacman:
            weights['distToPacman'] = -20
        else:
            weights['distToPacman'] = 0

        # If there is no visble enemy:
        if len(visibleGhost) == 0:
            # The pacman should try to get close to the food and capsule
            weights['distToFood'] = -20
            weights['distToCapsule'] = -5
            # It becomes more dangerous as the Pacman goes further
            # However, return home without food is meaningless
            weights['distToHome'] = 5 - 3 * successor.getAgentState(self.index).numCarrying
            # Enemy position is ignored
            weights['distToGhost'] = 0
            # To make the agent eat food, a huge weight should be assigned to 'foodCarrying'
            # Because once the agent eat a food, the distance to the nearest food will increase
            weights['foodCarrying'] = 1000
        # If there is ghost nearby:
        else:
            for enemy in visibleGhost:
                # If the enemy is not scared or the scared time is almost ended:
                if enemy.scaredTimer <= 5:
                    # Still try to get close to food, but less priority
                    # Concern more about the capsule and enemy distance
                    weights['distToFood'] = -5
                    weights['foodCarrying'] = 5
                    weights['distToCapsule'] = -15
                    weights['distToGhost'] = 20
                    # Return home gets higher priority
                    # More food the Pacman is carrying, more weight should be given to return
                    weights['distToHome'] = 5 * (-1 - successor.getAgentState(self.index).numCarrying)
                # If the enemy is scared
                else:
                    # Don't need to return
                    weights['distToHome'] = 0
                    # Ignore the ghost
                    weights['distToGhost'] = -10
                    # Ignore the capsule
                    weights['distToCapsule'] = 0
                    # Try to eat food
                    weights['distToFood'] = -10
                    weights['foodCarrying'] = 1000

        return weights



    def evaluate(self, gameState, action):
        features = self.getFeatures(gameState, action)
        weights = self.getWeights(gameState, action)
        evaluation = 0
        for feature, value in features.items():
            evaluation = evaluation + value * weights[feature]
        return evaluation

    def simulation(self, depth, gameState, decay):
        if depth == 0:
            simuResult = []
            actions = gameState.getLegalActions(self.index)
            actions.remove(Directions.STOP)
            reverse = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
            if reverse in actions and len(actions) > 1:
                actions.remove(reverse)
            action = random.choice(actions)
            newState = gameState.generateSuccessor(self.index, action)
            simuResult.append(self.evaluate(newState, Directions.STOP))
            return max(simuResult)
        else:
            simuResult = []
            actions = gameState.getLegalActions(self.index)
            reverse = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
            if reverse in actions and len(actions) > 1:
                actions.remove(reverse)
            for action in actions:
                newState = gameState.generateSuccessor(self.index, action)
                simuResult.append(
                    self.evaluate(newState, Directions.STOP) + decay * self.simulation(depth - 1, newState, decay))
            return max(simuResult)


    def chooseAction(self, gameState):
        actions = gameState.getLegalActions(self.agent.index)
        actions.remove(Directions.STOP)
        simuResult = []
        for action in actions:
            value = self.simulation(3, gameState.generateSuccessor(self.agent.index, action), 0.5)
            simuResult.append(value)
        maxResult = max(simuResult)
        bestActions = filter(lambda x: x[0] == maxResult, zip(simuResult, actions))
        chosenAction = random.choice(bestActions)[1]
        if self.preAction is None:
            self.preAction = chosenAction
        else:
            reverse = Directions.REVERSE[chosenAction]
            if self.preAction == reverse:
                self.trapTimer += 1
            else:
                self.trapTimer = 0
            self.preAction = chosenAction
        return chosenAction

class defenderAction():
    def __init__(self, agent, index, gameState):
        self.agent = agent
        self.index = index

        # Get the boundary
        if self.agent.red:
            boundary = (gameState.data.layout.width - 2) / 2
        else:
            boundary = ((gameState.data.layout.width - 2) / 2) + 1
        self.boundary = []
        for i in range(1, gameState.data.layout.height - 1):
            if not gameState.hasWall(boundary, i):
                self.boundary.append((boundary, i))

        self.preDefendingFood = []
        self.preEatenFood = None

    def getDenfendingTarget(self, gameState):
        defendingFood = self.agent.getFoodYouAreDefending(gameState).asList()
        targetFood = None
        # If no food has been eaten
        if self.preEatenFood is None:
            # Choose the food which is cloest to the boundary as the target
            foodDist = {}
            for food in defendingFood:
                minDistance = 1000
                for position in self.boundary:
                    distance = self.agent.getMazeDistance(food, position)
                    if distance < minDistance:
                        minDistance = distance
                foodDist[food] = minDistance
            targetFood = min(foodDist.items(), key=lambda x: x[1])[0]
        # If some food has been eaten
        else:
            minDistance = 1000
            for food in defendingFood:
                distance = self.agent.getMazeDistance(food, self.preEatenFood)
                if distance < minDistance:
                    minDistance = distance
                    targetFood = food
        return targetFood

    def getDenfendingPosition(self, gameState):
        minDistance = 1000
        target = self.getDenfendingTarget(gameState)
        defendPos = None
        for position in self.boundary:
            distance = self.agent.getMazeDistance(position, target)
            if distance < minDistance:
                minDistance = distance
                defendPos = position
        return defendPos

    def getFeatures(self, gameState, action):
        features = util.Counter()
        successor = gameState.generateSuccessor(self.index, action)
        position = successor.getAgentState(self.index).getPosition()
        defendPos = self.getDenfendingPosition(gameState)

        # Defender should keep at home
        if successor.getAgentState(self.index).isPacman:
            features['status'] = -1000
        else:
            features['status'] = 1000

        # Use the distance to the defending position as a target
        features['distToDefend'] = self.agent.getMazeDistance(defendPos, position)

        enemies = []
        for i in self.agent.getOpponents(gameState):
            enemies.append(successor.getAgentState(i))
        # Use the distance to nearest visible Pacman as a feature
        visiblePacman = []
        for enemy in enemies:
            if enemy.getPosition() != None and enemy.isPacman:
                visiblePacman.append(enemy)
        if len(visiblePacman) > 0:
            minDistToGhost = 6
            for ghost in visiblePacman:
                enemyPos = ghost.getPosition()
                enemyDistance = self.agent.getMazeDistance(position, enemyPos)
                if enemyDistance < minDistToGhost:
                    minDistToGhost = enemyDistance
            features['distToPacman'] = minDistToGhost
        else:
            # If no Pacman is visible, ignore the actual distance
            features['distToPacman'] = 0
        return features

    def getWeights(self, gameState, action):
        weights = util.Counter()

        weights['status'] = 1
        weights['distToDefend'] = -10
        weights['distToPacman'] = -30

        return weights


    def evaluate(self, gameState, action):
        features = self.getFeatures(gameState, action)
        weights = self.getWeights(gameState, action)
        evaluation = 0
        for feature, value in features.items():
            evaluation = evaluation + value * weights[feature]
        return evaluation

    def simulation(self, depth, gameState, decay):
        if depth == 0:
            simuResult = []
            actions = gameState.getLegalActions(self.index)
            actions.remove(Directions.STOP)
            reverse = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
            if reverse in actions and len(actions) > 1:
                actions.remove(reverse)
            for action in actions:
                newState = gameState.generateSuccessor(self.index, action)
                simuResult.append(self.evaluate(newState, Directions.STOP))
            return max(simuResult)
        else:
            simuResult = []
            actions = gameState.getLegalActions(self.index)
            reverse = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
            if reverse in actions and len(actions) > 1:
                actions.remove(reverse)
            for action in actions:
                newState = gameState.generateSuccessor(self.index, action)
                simuResult.append(
                    self.evaluate(newState, Directions.STOP) + decay * self.simulation(depth - 1, newState, decay))
            return max(simuResult)

    def chooseAction(self, gameState):
        actions = gameState.getLegalActions(self.agent.index)
        actions.remove(Directions.STOP)
        simuResult = []
        for action in actions:
            value = self.simulation(3, gameState.generateSuccessor(self.agent.index, action), 0.5)
            simuResult.append(value)
        maxResult = max(simuResult)
        bestActions = filter(lambda x: x[0] == maxResult, zip(simuResult, actions))
        chosenAction = random.choice(bestActions)[1]
        if len(self.preDefendingFood) == 0:
            self.preDefendingFood = self.agent.getFoodYouAreDefending(gameState).asList()
            return chosenAction
        else:
            if len(self.preDefendingFood) == len(self.agent.getFoodYouAreDefending(gameState).asList()):
                return chosenAction
            else:
                for food in self.preDefendingFood:
                    if food not in self.agent.getFoodYouAreDefending(gameState).asList():
                        self.preEatenFood = food
                self.preDefendingFood = self.agent.getFoodYouAreDefending(gameState).asList()
                return chosenAction







##########
# Agents #
##########

class Attacker(CaptureAgent):

    def __init__(self, index):
        CaptureAgent.__init__(self, index)

    def registerInitialState(self, gameState):
        """
        This method handles the initial setup of the
        agent to populate useful fields (such as what team
        we're on).

        A distanceCalculator instance caches the maze distances
        between each pair of positions, so your agents can use:
        self.distancer.getDistance(p1, p2)

        IMPORTANT: This method may run for at most 15 seconds.
        """

        '''
        Make sure you do not delete the following line. If you would like to
        use Manhattan distances instead of maze distances in order to save
        on initialization time, please take a look at
        CaptureAgent.registerInitialState in captureAgents.py.
        '''
        CaptureAgent.registerInitialState(self, gameState)

        '''
        Your initialization code goes here, if you need any.
        '''
        self.attackStatus = attackerAction(self, self.index, gameState)


    def chooseAction(self, gameState):

        return self.attackStatus.chooseAction(gameState)

class Defender(CaptureAgent):
    def __init__(self, index):
        CaptureAgent.__init__(self, index)

    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)
        self.defendStatus = defenderAction(self, self.index, gameState)

    def chooseAction(self, gameState):

        return self.defendStatus.chooseAction(gameState)