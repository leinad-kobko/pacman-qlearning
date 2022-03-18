# featureExtractors.py
# --------------------
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


"Feature extractors for Pacman game states"

from smtplib import SMTPResponseException
from game import Directions, Actions
import util

class FeatureExtractor:
    def getFeatures(self, state, action):
        """
          Returns a dict from features to counts
          Usually, the count will just be 1.0 for
          indicator functions.
        """
        util.raiseNotDefined()

class IdentityExtractor(FeatureExtractor):
    def getFeatures(self, state, action):
        feats = util.Counter()
        feats[(state,action)] = 1.0
        return feats

class CoordinateExtractor(FeatureExtractor):
    def getFeatures(self, state, action):
        feats = util.Counter()
        feats[state] = 1.0
        feats['x=%d' % state[0]] = 1.0
        feats['y=%d' % state[0]] = 1.0
        feats['action=%s' % action] = 1.0
        return feats

def closestFood(pos, food, walls):
    """
    closestFood -- this is similar to the function that we have
    worked on in the search project; here its all in one place
    """
    fringe = [(pos[0], pos[1], 0)]
    expanded = set()
    while fringe:
        pos_x, pos_y, dist = fringe.pop(0)
        if (pos_x, pos_y) in expanded:
            continue
        expanded.add((pos_x, pos_y))
        # if we find a food at this location then exit
        if food[pos_x][pos_y]:
            return dist
        # otherwise spread out from the location to its neighbours
        nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
        for nbr_x, nbr_y in nbrs:
            fringe.append((nbr_x, nbr_y, dist+1))
    # no food found
    return None

class SimpleExtractor(FeatureExtractor):
    """
    Returns simple features for a basic reflex Pacman:
    - whether food will be eaten
    - how far away the next food is
    - whether a ghost collision is imminent
    - whether a ghost is one step away
    """

    def getFeatures(self, state, action):
        # extract the grid of food and wall locations and get the ghost locations
        food = state.getFood()
        walls = state.getWalls()
        ghosts = state.getGhostPositions()

        features = util.Counter()

        features["bias"] = 1.0

        # compute the location of pacman after he takes the action
        x, y = state.getPacmanPosition()
        dx, dy = Actions.directionToVector(action)
        next_x, next_y = int(x + dx), int(y + dy)

        # count the number of ghosts 1-step away
        features["#-of-ghosts-1-step-away"] = sum((next_x, next_y) in Actions.getLegalNeighbors(g, walls) for g in ghosts)

        # if there is no danger of ghosts then add the food feature
        if not features["#-of-ghosts-1-step-away"] and food[next_x][next_y]:
            features["eats-food"] = 1.0

        dist = closestFood((next_x, next_y), food, walls)
        if dist is not None:
            # make the distance a number less than one otherwise the update
            # will diverge wildly
            features["closest-food"] = float(dist) / (walls.width * walls.height)
        features.divideAll(10.0)
        return features

class myExtractor(FeatureExtractor):
    """
    Contains the same features as SimpleExtractor,
    with the addition of:
        - eat capsule
        - avoid capsule
        - closest capsule
        - eat scared ghost
        - closest scared ghost
        - closest angry ghost
        - can't stop
    """
    
    def getFeatures(self, state, action):
        # Extract food, wall, ghost and capsule locations from the grid
        food = state.getFood()
        walls = state.getWalls()
        ghostStates = state.getGhostStates()
        capsules = []

        # Categorize the ghosts
        scaredGhosts = []
        angryGhosts = []
        for ghost in ghostStates:
            if not ghost.scaredTimer:
                angryGhosts.append(ghost)
            else:
                scaredGhosts.append(ghost)
        
        # Compute the location of pacman after he takes the action
        x, y = state.getPacmanPosition()
        dx, dy = Actions.directionToVector(action)
        next_x, next_y = int(x + dx), int(y + dy)
        
        # Gather capsule information if there are angry ghosts
        if angryGhosts:
            capsules = state.getCapsules()

        features = util.Counter()
        features["bias"] = 1.0

        wallArea = (walls.width * walls.height)

        # Count the number of angry ghosts 1-step away
        features["#-of-angry-ghosts-1-step-away"] = sum((next_x, next_y) in Actions.getLegalNeighbors(g.configuration.pos, walls) for g in angryGhosts)

        # Eat scared ghosts if no angry ghosts are in the way
        if scaredGhosts and \
            not features["#-of-angry-ghosts-1-step-away"] and \
            (next_x, next_y) in getGhostPositions(scaredGhosts):
            features["eats-ghost"] = 1.0
        # Chase scared ghosts if close enough
        scaredGhostDist = closestGhost((next_x, next_y), scaredGhosts, walls)
        if scaredGhostDist is not None and \
            not ghostNearby((next_x, next_y), angryGhosts, walls, 2) and \
            state.getNumFood() > 5:
            features["closest-scared-ghost"] = 1.0 - (float(scaredGhostDist) / wallArea)

        # Pacman shouldn't stand still!
        if (action != Directions.STOP):
            features["can't-stop"] = 1.0

        # Determine whether to go for a capsule
        if capsules and \
            not features["#-of-angry-ghosts-1-step-away"] and \
            (next_x, next_y) in capsules:
            features["eats-capsule"] = 1.0
        # Chase the nearest capsule while there are no scared ghosts
        capsuleDist = closestCapsule((next_x, next_y), capsules, walls)
        if capsuleDist is not None and \
            not scaredGhosts:
            features["closest-capsule"] = float(capsuleDist) / wallArea

        # Determine whether to go for food
        if not capsules and \
            not features["#-of-angry-ghosts-1-step-away"] and \
            food[next_x][next_y]:
            features["eats-food"] = 1.0
        # If there's nothing else to chase, chase food
        foodDist = closestFood((next_x, next_y), food, walls)
        if not capsules and foodDist is not None:
            # Avoid capsules while all ghosts are scared
            if (next_x, next_y) not in state.getCapsules():
                features["avoid-capsule"] = 1.0
            features["closest-food"] = float(foodDist) / wallArea

        # Pacman should favour food thats further from angry ghosts 
        angryGhostDist = closestGhost((next_x, next_y), angryGhosts, walls)
        if angryGhosts and \
            angryGhostDist is not None and \
            features["closest-food"]:
            features["closest-angry-ghost"] = features["closest-food"] + (1.0 - (float(angryGhostDist) / wallArea))

        features.divideAll(10.0)
        
        return features

# This function is based of the closestFood function provided.
# returns the step distance of the closest entity from entities
def closestOfEntity(pos, entities, walls):
    if not entities:
        return None
    
    fringe = [(pos[0], pos[1], 0)]
    expanded = set()
    while fringe:
        pos_x, pos_y, dist = fringe.pop(0)
        if (pos_x, pos_y) in expanded:
            continue
        expanded.add((pos_x, pos_y))
        # if we find the entity at this 
        # location return the step distance
        if (pos_x, pos_y) in entities:
            return dist
        # otherwise spread out from the location to its neighbours
        nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
        for nbr_x, nbr_y in nbrs:
            fringe.append((nbr_x, nbr_y, dist+1))
    
    return None

# Returns the distance in steps of the closest capsule
def closestCapsule(pos, capsules, walls):
    return closestOfEntity(pos, capsules, walls)

# Returns the distance in steps of the closest ghost
def closestGhost(pos, ghosts, walls):
    return closestOfEntity(pos, getGhostPositions(ghosts), walls)

# Returns true if a ghost from ghosts is within maxDist
def ghostNearby(pos, ghosts, walls, maxDist):
    if not ghosts:
        return False
    ghostDist = closestGhost(pos, ghosts, walls)
    if ghostDist == None or ghostDist < maxDist:
        return True
    return False

# Get the list of (x, y) positions from ghosts
def getGhostPositions(ghosts):
    positions = []
    for ghost in ghosts:
        x, y = ghost.getPosition()
        dx, dy = Actions.directionToVector(ghost.getDirection())
        positions.append((int(x+dx), int(y+dy)))
    return positions