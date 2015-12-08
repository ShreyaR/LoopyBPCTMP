__author__ = 'shreyarajpal'

from math import log10, log, exp, fabs
import time
from helperFunctions import ocrDat, transDat, getPairSkipFactor, getSkipFactor, findingSkips, assignmentToNumber, numberToAssignment

#Returns Bethe Cluster Graph (as dictionary), along with mapping of each cluster index to its underlying factor
def getBetheClusterGraph(w1, w2, model):
    graph = {x: [] for x in xrange(len(w1) + len(w2))}
    #graph = {}

    clusterMapping = {} #Gives info on which factor index is actually what
    reverseClusterMapping = {}

    numOfFactors = 0
    sk1, sk2, ps = findingSkips(w1, w2)
    if model>=1:
        for i in xrange(len(w1) + len(w2)):
            graph[i].append(('o', i))
            clusterMapping[('o', i)] = numOfFactors
            reverseClusterMapping[numOfFactors] = ('o', i)
            numOfFactors += 1
    if model>=2:
        for i in xrange(len(w1) - 1):
            graph[i].append(('t', (i, i+1)))
            graph[i+1].append(('t', (i, i+1)))
            clusterMapping[('t', (i, i+1))] = numOfFactors
            reverseClusterMapping[numOfFactors] = ('t', (i, i+1))
            numOfFactors += 1
        for i in xrange(len(w2) - 1):
            graph[len(w1) + i].append(('t', (len(w1) + i,len(w1) +  i+1)))
            graph[len(w1) + i+1].append(('t', (len(w1) + i,len(w1) +  i+1)))
            clusterMapping[('t', (len(w1) + i,len(w1) +  i+1))] = numOfFactors
            reverseClusterMapping[numOfFactors] = ('t', (len(w1) + i,len(w1) +  i+1))
            numOfFactors += 1
    if model>=3:
        for skipFactor in sk1:
            i,j = skipFactor
            graph[i].append(('s', (i,j)))
            graph[j].append(('s', (i,j)))
            clusterMapping[('s', (i, j))] = numOfFactors
            reverseClusterMapping[numOfFactors] = ('s', (i, j))
            numOfFactors += 1
        for skipFactor in sk2:
            i,j = skipFactor
            i += len(w1)
            j += len(w1)
            graph[i].append(('s', (i,j)))
            graph[j].append(('s', (i,j)))
            clusterMapping[('s', (i, j))] = numOfFactors
            reverseClusterMapping[numOfFactors] = ('s', (i, j))
            numOfFactors += 1
    if model==4:
        for pairSkip in ps:
            i,j = pairSkip
            graph[i].append(('p', (i,len(w1) + j)))
            graph[len(w1) + j].append(('p', (i,len(w1) + j)))
            clusterMapping[('p', (i,len(w1) +  j))] = numOfFactors
            reverseClusterMapping[numOfFactors] = ('p', (i,len(w1) +  j))
            numOfFactors += 1

    clusterGraph = {x: set() for x in xrange(numOfFactors)} #Gives info on which clusters are connected to each other via which separator

    #print graph, '\n\n'
    #print reverseClusterMapping, '\n\n'
    #print clusterMapping, '\n\n'

    for i in graph.keys():
        factors = graph[i]
        #for j in xrange(len(factors)):
        factor1 = factors[0]
        index1 = clusterMapping[factor1]
        for k in range(1, len(factors)):
            factor2 = factors[k]
            index2 = clusterMapping[factor2]
            clusterGraph[index1].add((index2,i))
            clusterGraph[index2].add((index1,i))

    return clusterGraph, reverseClusterMapping

#Returns marginalized beliefs
def marginalize(belief, separator, varsInBelief, typeOfFactor):
    if typeOfFactor=='o':
        return belief
    else:
        tempMaxFinder = {x:[] for x in xrange(10)}
        msg = {x:0 for x in xrange(10)}
        n = int(log10(len(belief.keys())))
        for i in belief.keys():
            assignment = numberToAssignment(i,n)
            assignmentToSep = assignment[varsInBelief.index(separator)]
            tempMaxFinder[assignmentToNumber(assignmentToSep)].append(belief[i])

        for i in tempMaxFinder.keys():
            msg[i] = max(tempMaxFinder[i])
        return msg

#Returns Updated Belief
def updateBelief(oldBelief, msg, separator, varsInBelief, edgeBelief, typeOfFactor):
    if typeOfFactor=='o':
        newBelief = {}
        for i in oldBelief.keys():
            newBelief[i] = oldBelief[i] + msg[i] - edgeBelief[i]
        return newBelief

    else:
        n = int(log10(len(oldBelief.keys())))
        newBelief = {}
        for i in oldBelief.keys():
            assignment = numberToAssignment(i, n)
            assignmentToSeparator = assignment[varsInBelief.index(separator)]
            indexSeparator = assignmentToNumber(assignmentToSeparator)

            newBelief[i] = oldBelief[i] + msg[indexSeparator] - edgeBelief[indexSeparator]

        return newBelief


def beliefUpdate(w1, w2, model):
    betheGraph, betheMapping = getBetheClusterGraph(w1, w2, model)

    #TODO: Check if two clusters can be connected by more than one edge (and therefore separator). Example, trans edge and skip edge b/w letters.
    edgesInBetheGraph = {}
    numOfEdges = 0
    for i in betheGraph.keys():
        neighours = list(betheGraph[i])
        for j in neighours:
            edge = ({i,j[0]}, j[1])
            if edge not in edgesInBetheGraph.values():
                edgesInBetheGraph[numOfEdges] = edge
                numOfEdges += 1

    #print edgesInBetheGraph
    #print betheMapping

    ##############################################################################################
    #Cluster Graph Initialization

    #print betheGraph
    #print betheMapping
    #print edgesInBetheGraph
    #print '\n\n'

    beliefAtEachNode = {x:{} for x in betheMapping.keys()}
    beliefAtEachEdge = {x:{y:0 for y in xrange(10)} for x in edgesInBetheGraph.keys()}

    for x in betheMapping.keys():
        factor = betheMapping[x]
        typeOfFactor = factor[0]
        varsInFactor = factor[1]

        if typeOfFactor == 'o':
            if varsInFactor<len(w1):
                letter = w1[varsInFactor]
            else:
                letter = w2[varsInFactor - len(w1)]
            beliefAtEachNode[x] = ocrDat[letter].copy()
            beliefAtEachNode[x] = {assignmentToNumber(t): beliefAtEachNode[x][t] for t in beliefAtEachNode[x].keys()}

        elif typeOfFactor == 't':
            for i in xrange(100):
                assignment = numberToAssignment(i,2)
                beliefAtEachNode[x][i] = transDat[assignment[0]][assignment[1]]

        elif typeOfFactor == 's':
            for i in xrange(100):
                assignment = numberToAssignment(i,2)
                beliefAtEachNode[x][i] = getSkipFactor(assignment[0], assignment[1])

        else:
            for i in xrange(100):
                assignment = numberToAssignment(i,2)
                beliefAtEachNode[x][i] = getPairSkipFactor(assignment[0], assignment[1])

    ##############################################################################################
    #Initialization Complete. Now starting Belief Updates.

    numOfIterations = 0
    start = time.clock()
    while(True):
        maxError = -50000
        oldBelief = beliefAtEachNode.copy()

        for i in edgesInBetheGraph.keys():
            node1,node2 = edgesInBetheGraph[i][0]
            separator = edgesInBetheGraph[i][1]
            typeOfFactor1 = betheMapping[node1][0]
            typeOfFactor2 = betheMapping[node2][0]
            varsInNode1 = betheMapping[node1][1]
            varsInNode2 = betheMapping[node2][1]

            msg1to2 = marginalize(beliefAtEachNode[node1], separator, varsInNode1, typeOfFactor1)
            beliefAtEachNode[node2] = updateBelief(beliefAtEachNode[node2], msg1to2, separator, varsInNode2, beliefAtEachEdge[i], typeOfFactor2)
            beliefAtEachEdge[i] = msg1to2

            msg2to1 = marginalize(beliefAtEachNode[node2], separator, varsInNode2, typeOfFactor2)
            beliefAtEachNode[node1] = updateBelief(beliefAtEachNode[node1], msg2to1, separator, varsInNode1, beliefAtEachEdge[i], typeOfFactor1)
            beliefAtEachEdge[i] = msg2to1

            error1 = max([fabs(beliefAtEachNode[node1][i] - oldBelief[node1][i]) for i in oldBelief[node1].keys()])
            error2 = max([fabs(beliefAtEachNode[node2][i] - oldBelief[node2][i]) for i in oldBelief[node2].keys()])
            error = max(error1, error2)

            #print error

            if maxError<error:
                maxError = error

        ##############################################################################################
        #Normalization
        for i in beliefAtEachNode.keys():
            z = 0
            for x in beliefAtEachNode[i].keys():
                z += exp(beliefAtEachNode[i][x])
            z = log(z)
            for x in beliefAtEachNode[i].keys():
                beliefAtEachNode[i][x] -= z

        for i in beliefAtEachEdge.keys():
            z = 0
            for x in beliefAtEachEdge[i].keys():
                z += exp(beliefAtEachEdge[i][x])
            z = log(z)
            for x in beliefAtEachEdge[i].keys():
                beliefAtEachEdge[i][x] -= z

        numOfIterations += 1
        #print 'numOfIterations',numOfIterations
        #print 'maxError', maxError

        if (maxError<0.0001) or (numOfIterations >= 50):
            end = time.clock()
            duration = end - start
            return beliefAtEachNode, betheMapping, duration
            break


#a,b = getBetheClusterGraph([405, 840, 840], [826, 623, 840], 4)
#print a, b

'''
def getPartitionFunction(w1, w2, model):
    marginalAtCluster = beliefUpdate(w1, w2, model)[0]
    print marginalAtCluster

    commonFactor = min(marginalAtCluster.values())
    sum = 0
    for i in marginalAtCluster.keys():
        sum += exp(marginalAtCluster[i] - commonFactor)

    sum *= exp(commonFactor)
    return sum
'''

def getMaxMarginalAssignment(w1, w2, model):
    belief, betheMapping, duration = beliefUpdate(w1, w2, model)

    #print belief
    #print betheMapping

    a1 = ''

    for i in xrange(len(w1)):
        maxB = -500
        maxAss = -1
        for x in belief[i].keys():
            if belief[i][x] > maxB:
                maxB = belief[i][x]
                maxAss = x

        maxAss = numberToAssignment(maxAss,1)[0]
        a1 += maxAss


    a2 = ''
    for i in xrange(len(w2)):
        maxB = -500
        maxAss = -1
        for x in belief[i + len(w1)].keys():
            if belief[i + len(w1)][x] > maxB:
                maxB = belief[i + len(w1)][x]
                maxAss = x
        maxAss = numberToAssignment(maxAss,1)[0]
        a2 += maxAss

    return a1,a2, duration


def getLogLikelihood(w1, w2, a1, a2, model):
    belief, ran, duration = beliefUpdate(w1, w2, model)

    LL = 0

    for i in xrange(len(w1)):
        assignedLetter = a1[i]
        LL += belief[i][assignmentToNumber(assignedLetter)]

    for i in xrange(len(w2)):
        assignedLetter = a2[i]
        LL += belief[i + len(w1)][assignmentToNumber(assignedLetter)]

    return LL, duration


def getMAPassignment(w1, w2, model):
    belief, mapping, duration = beliefUpdate(w1, w2, model)

    a1 = ''
    a2 = ''

    for i in xrange(len(w1)):
        relevantBelief = belief[i]
        maxProb = max(relevantBelief.values())

        for j in relevantBelief.keys():
            if relevantBelief[j] ==maxProb:
                maxAss = j

        a1 += numberToAssignment(maxAss,1)[0]


    for i in xrange(len(w2)):
        relevantBelief = belief[i + len(w1)]
        maxProb = max(relevantBelief.values())

        for j in relevantBelief.keys():
            if relevantBelief[j] ==maxProb:
                maxAss = j

        a2 += numberToAssignment(maxAss,1)[0]


    return a1, a2, duration
#x =  beliefUpdate([711,438,883,455,100,438],[392,438,266,584,455,600], 4)
#x = beliefUpdate([405, 840, 840], [826, 623, 840], 4)
#print x

#print assignmentToNumber(['a', 'a'])
#print numberToAssignment(910, 3)

#print getMaxMarginalAssignment([405, 840, 840], [826, 623, 840], 4)
