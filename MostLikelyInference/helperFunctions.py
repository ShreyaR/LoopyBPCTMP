__author__ = 'shreyarajpal'
from math import log, exp
from collections import deque
import time

#global ocrDat, transDat
characterArray = ['e','t','a','o','i','n','s','h','r','d']
ocrDat = []
transDat = {}

def readOCRPotentials():
    global ocrDat
    ocrDat = [{} for _ in xrange(1000)]#*numOfImages
    ocrInfo = open('data/ocr.dat', 'r')
    for line in ocrInfo:
        info = line.rstrip().split('\t')
        ocrDat[int(info[0])][info[1]] = log(float(info[2]))
    ocrInfo.close()
    return

def readTransPotentials():
    global transDat
    transInfo = open('data/trans.dat', 'r')

    for item in characterArray:
        transDat[item] = {}
        for i in characterArray:
            transDat[item][i]=-1

    for line in transInfo:
        info = line.rstrip().split('\t')
        transDat[info[0]][info[1]] = log(float(info[2]))

    transInfo.close()
    return

def getSkipFactor(a1, a2):
    if (a1==a2):
        return log(5)
    else:
        return 0

def getPairSkipFactor(a1, a2):
    if a1==a2:
        return log(5)
    else:
        return 0

def assignmentToNumber(a):
    lenOfWord = len(a)
    index = 0
    for i in xrange(lenOfWord):
        multiplier = characterArray.index(a[i])
        index += multiplier * (pow(10, i))
    return index

def numberToAssignment(x, l):
    a = []
    i = 1
    for j in xrange(l):
        if x:
            a.append(x%10)
            x=x/10
        else:
            a.append(0)

    b = [characterArray[t] for t in a]
    return b

#Returns 3 lists -> skip edges in w1, skip edges in w2 and pair skip edges
def findingSkips(w1, w2):


    skipEdge1 = []
    skipEdge2 = []
    pairSkip = []

    n1 = len(w1)
    n2 = len(w2)

    #Finding Skip Edge in Word 1
    for i in xrange(n1):
        for j in range(i+1, n1):
            if w1[i]==w1[j]:
                skipEdge1.append((i,j))

    #Finding Skip Edge in Word 2
    for i in xrange(n2):
        for j in range(i+1, n2):
            if w2[i]==w2[j]:
                skipEdge2.append((i,j))

    #Finding Pair Skip Edges
    for i in xrange(n1):
        for j in xrange(n2):
            if w1[i]==w2[j]:
                pairSkip.append((i, j))

    return skipEdge1, skipEdge2, pairSkip

#Returns undirected graph as a dictionary
def getGraph(n1, n2, sk1, sk2, ps, model):
    graph = {}

    #Graph created has nodes from 0 to (n1 + n2 - 1). First n1 nodes are nodes in w1, next n2 are nodes in w2.
    if model>= 1:
        for i in xrange(n1 + n2):
            graph[i] = set()
        '''
        for i in xrange(n1):
            if i>0 and i<(n1 - 1):
                graph[i] = {i-1, i+1}
            elif i==0:
                graph[i] = {i+1}
            else:
                graph[i] = {i-1}
        '''
    if model>=2:
        '''
        for i in range(n1, n1+n2):
            if i>n1 and i<(n1 + n2 - 1):
                graph[i] = {i-1, i+1}
            elif i==n1:
                graph[i] = {i+1}
            else:
                graph[i] = {i-1}
        '''
        for i in xrange(n1 - 1):
            graph[i].add(i+1)
            graph[i+1].add(i)
        for i in xrange(n2 - 1):
            graph[n1 + i].add(n1 + i+1)
            graph[n1 + i+1].add(n1 + i)
    if model>=3:
        for i in sk1:
            graph[i[0]].add(i[1])
            graph[i[1]].add(i[0])
        for i in sk2:
            graph[n1 + i[0]].add(n1 + i[1])
            graph[n1 + i[1]].add(n1 + i[0])
    if model==4:
        for i in ps:
            graph[i[0]].add(n1 + i[1])
            graph[n1 + i[1]].add(i[0])

    return graph

#Returns ordering as a list (of length n1 + n2)
def orderingVE(n1, n2, sk1, sk2, ps, model):

    #Constructing the Graph
    graph = getGraph(n1, n2, sk1, sk2, ps, model)
    #print graph

    #Min Fill Edge Greedy Search
    Marked = set()
    Unmarked = set(graph.keys())

    #Find edge that introduces least number of fill edges:
    ordering = []
    for count in xrange(len(graph.keys())):
        #print 'Count = ',str(count),'\n','\n'
        minFillEdges = 100
        minFillNode = -1
        for i in graph.keys():
            #print 'Node ',str(i)
            fillEdges = 0
            neighbors = list(graph[i])
            numOfNeighbors = len(neighbors)

            for x in xrange(numOfNeighbors):
                for y in range(x+1, numOfNeighbors):
                    neighborX = neighbors[x]
                    neighborY = neighbors[y]
                    #print 'X = ',neighborX
                    #print 'Y = ',neighborY
                    if neighborY not in graph[neighborX]:
                        fillEdges += 1

            if fillEdges < minFillEdges:
                minFillEdges = fillEdges
                minFillNode = i

            #print minFillEdges, minFillNode

        #print 'MinFillNode', minFillNode, 'MinFillEdges',minFillEdges

        ordering.append(minFillNode)
        #print 'ordering',ordering
        Marked.add(minFillNode)
        Unmarked.remove(minFillNode)
        neighbors = list(graph[minFillNode])
        #print graph
        #print 'Neighbors', neighbors
        numOfNeighbors = len(neighbors)
        for x in xrange(numOfNeighbors):
            neighborX = neighbors[x]
            graph[neighborX].remove(minFillNode)
            for y in range(x+1, numOfNeighbors):
                neighborY = neighbors[y]
                if neighborY not in graph[neighborX]:
                    graph[neighborX].add(neighborY)
                    graph[neighborY].add(neighborX)
        #print graph
        del graph[minFillNode]
        #print graph

    return ordering

#Returns an ordering (in a nice format: optional); acts as a wrapper around 'findingSkips' and 'orderingVE'
def getOrdering(w1, w2, model):
    sk1, sk2, ps = findingSkips(w1, w2)
    n1 = len(w1)
    n2 = len(w2)
    ordering = orderingVE(n1, n2, sk1, sk2, ps, model)

    return ordering

#Returns the clique tree after VE
def getCliqueTree(w1, w2, model):
    cliqueMapping = {}
    tree = {}

    n1 = len(w1)
    n2 = len(w2)
    sk1, sk2, ps = findingSkips(w1, w2)
    graph = getGraph(n1, n2, sk1, sk2, ps, model)
    ordering = getOrdering(w1, w2, model)
    #print 'ORDERING', ordering, '\n\n'

    count = 0

    sepSets = []

    for i in ordering:
        #Basic book-keeping
        clique = graph[i]
        clique.add(i)
        #print clique
        cliqueMapping[count] = clique
        #print cliqueMapping
        tree[count] = set()
        #print tree
        clique.remove(i)
        neighbors = list(clique)

        numOfNeighbors = len(neighbors)

        #Graph modification to reflect VE
        for x in xrange(numOfNeighbors):
            Xneighbor = neighbors[x]
            graph[Xneighbor].remove(i)
            for y in range(x+1, numOfNeighbors):
                Yneighbor = neighbors[y]
                if Yneighbor not in graph[Xneighbor]:
                    graph[Xneighbor].add(Yneighbor)
                    graph[Yneighbor].add(Xneighbor)

        #Adding edges to the Clique Tree via sepSets
        separatorsToBeRemoved = []
        for t in sepSets:
            separator = t[0]
            associatedClique = t[1]
            if i in separator:
                tree[associatedClique].add(count)
                tree[count].add(associatedClique)
                separatorsToBeRemoved.append(t)

        for t in separatorsToBeRemoved:
            sepSets.remove(t)

        #Book-keeping for maintaining sepSets
        sepSets.append((set(neighbors), count))

        count += 1

    for i in cliqueMapping.keys():
        cliqueMapping[i].add(ordering[i])

    return tree, cliqueMapping

#Returns the marginals over every node
def forwardBackward(w1, w2, model):
    cliqueTree, mapping = getCliqueTree(w1, w2, model)
    mapping = {i: list(mapping[i]) for i in mapping.keys()}

    parentTree = {x: set() for x in xrange(len(w1) + len(w2))}
    childrenTree = {x: set() for x in xrange(len(w1) + len(w2))}

    #Designate one node as root
    root = cliqueTree.keys()[0]

    #Designate parents and children for each node via DFS
    queue = deque([root])
    while(len(queue)):
        item = queue.popleft()#Gives you a node in the clique tree
        childrenTree[item] = cliqueTree[item]
        for i in cliqueTree[item]:
            parentTree[i].add(item)
            cliqueTree[i].remove(item)
            queue.append(i)

    ##############################################################################################

    #Asign factors to each node in the clique-tree
    sk1, sk2, ps = findingSkips(w1, w2)
    factors = {x:set() for x in cliqueTree.keys()}
    if model>=1:
        #Associate each OCR factor with a clique
        for i in xrange(len(w1) + len(w2)):
            for j in xrange(len(mapping.keys())):
                if i in mapping[j]:
                    factors[j].add(('o',i))
                    break

    if model>=2:
        #Associate each transition factor with a clique
        for i in xrange(len(w1) - 1):
            for j in xrange(len(mapping.keys())):
                if (i in mapping[j]) and ((i + 1) in mapping[j]):
                    factors[j].add(('t',(i, i + 1)))
                    break
        for i in xrange(len(w2) - 1):
            for j in xrange(len(mapping.keys())):
                if ((i + len(w1)) in mapping[j]) and ((i + 1 + len(w1)) in mapping[j]):
                    factors[j].add(('t',(i + len(w1), i + 1 + len(w1))))
                    break

    if model>=3:
        #Associate each skip factor with a clique:
        for item in sk1:
            l1,l2 = item
            for j in xrange(len(mapping.keys())):
                cluster = mapping[j]
                if (l1 in cluster) and (l2 in cluster):
                    factors[j].add(('s', (l1, l2)))
                    break
        for item in sk2:
            l1,l2 = item
            l1 += len(w1)
            l2 += len(w1)
            for j in xrange(len(mapping.keys())):
                cluster = mapping[j]
                if (l1 in cluster) and (l2 in cluster):
                    factors[j].add(('s', (l1, l2)))
                    break

    if model==4:
        #Associate every pair skip factor with a clique
        for item in ps:
            l1,l2 = item
            l2 += len(w1)
            for j in xrange(len(mapping.keys())):
                cluster = mapping[j]
                if (l1 in cluster) and (l2 in cluster):
                    factors[j].add(('p',(l1, l2)))
                    break

    ##############################################################################################

    ready = set()

    #Factor Repo is a large repository storing all factors. At index i is a factor table
    factorRepo = []
    #Factor Mapping contains the mapping of indices in factor-repo to their underlying variables.
    factorMapping = {}
    #FactorStorage maps the indices in factor repo to the separator on which they were passed
    separatorMemoized = {}
    count = 0

    marked = {x for x in childrenTree if childrenTree[x]}

    queue = deque([])
    [queue.append(x) for x in childrenTree.keys() if not childrenTree[x]]

    marginalAtEachNode = {}

    MAP = {x:0 for x in xrange(len(w1) + len(w2))}

    start = time.clock()
    while(len(queue)):
        #print queue
        item = queue.popleft()
        #print item
        if parentTree[item]:
            parent = True
            itemsInParentNode = mapping[list(parentTree[item])[0]]
        else:
            parent = False
            itemsInParentNode = set()

        itemsInCurrentNode = mapping[item]
        termsInMessage = list(set(itemsInCurrentNode).intersection(set(itemsInParentNode)))

        #Create Message
        termsInMessage = list(termsInMessage)
        factorMapping[count] = termsInMessage
        if parent:
            separatorMemoized[(item, list(parentTree[item])[0])] = count
        factorRepo.append({})
        itemsInCurrentNode = list(itemsInCurrentNode)

        #Create the 'Psi' factor for summing out a node
        psiFactor = {}
        for i in xrange(pow(10,len(itemsInCurrentNode))):
            assignment = numberToAssignment(i, len(itemsInCurrentNode))
            potential = 0
            for x in factors[item]:
                typeOfFactor = x[0]
                if typeOfFactor=='o':
                    relevantVariable = x[1]
                    if relevantVariable<len(w1):
                        rv = w1[relevantVariable]
                    else:
                        rv = w2[relevantVariable - len(w1)]
                    place = itemsInCurrentNode.index(relevantVariable)
                    potential += ocrDat[rv][assignment[place]]
                elif typeOfFactor=='t':
                    fromVariable, toVariable = x[1]
                    fromPlace = itemsInCurrentNode.index(fromVariable)
                    toPlace = itemsInCurrentNode.index(toVariable)
                    potential += transDat[assignment[fromPlace]][assignment[toPlace]]
                elif typeOfFactor=='s':
                    var1, var2 = x[1]
                    place1 = itemsInCurrentNode.index(var1)
                    place2 = itemsInCurrentNode.index(var2)
                    potential += getSkipFactor(assignment[place1],assignment[place2])
                elif typeOfFactor=='p':
                    var1, var2 = x[1]
                    place1 = itemsInCurrentNode.index(var1)
                    place2 = itemsInCurrentNode.index(var2)
                    potential += getPairSkipFactor(assignment[place1],assignment[place2])
                else:
                    index = x[1]
                    variables = factorMapping[index]
                    tempAssignment = [assignment[itemsInCurrentNode.index(i)] for i in variables]
                    lookUpEntry = assignmentToNumber(tempAssignment)
                    potential += factorRepo[index][lookUpEntry]
            psiFactor[assignmentToNumber(assignment)] = potential

        marginalAtEachNode[item] = psiFactor

        #Create the 'tau' factor after eliminating
        #TODO: Check if this is correct
        if (len(itemsInCurrentNode) - len(termsInMessage)):
            tauFactor = {t: 0 for t in xrange(pow(10, len(termsInMessage)))}
        else:
            tauFactor = {t: 0 for t in xrange(pow(10, len(termsInMessage)))}

        tempMaxFinder = {x: [] for x in xrange(pow(10,len(termsInMessage)))}

        #print tauFactor
        for i in psiFactor.keys():
            assignment = numberToAssignment(i, len(itemsInCurrentNode))
            assignmentOfSeparator = [assignment[itemsInCurrentNode.index(x)] for x in termsInMessage]
            index = assignmentToNumber(assignmentOfSeparator)

            #print itemsInCurrentNode,termsInMessage, i, index, assignment, assignmentOfSeparator

            tempMaxFinder[index].append(psiFactor[i])
            #tauFactor[index] += exp(psiFactor[i])

        for i in tempMaxFinder.keys():
            tauFactor[i] = max(tempMaxFinder[i])

        #tauFactor = {x:log(tauFactor[x]) for x in tauFactor.keys()}
        factorRepo[count] = tauFactor

        if parent:
            factors[list(parentTree[item])[0]].add(('m', count))

        ready.add(item)

        toBeRemovedFromMarked = set()

        for k in marked:
            allChildren = childrenTree[k]
            flag = True
            for l in allChildren:
                if l not in ready:
                    flag = False
            if flag:
                queue.append(k)
                toBeRemovedFromMarked.add(k)


        marked = marked.difference(toBeRemovedFromMarked)
        count += 1

    #print separatorMemoized

    #print marginalAtEachNode
    ##############################################################################################
    #Forward Message Passing Completed. Backward message passing hereon.


    backwardsQueue = deque([root])

    while(len(backwardsQueue)):
        item = backwardsQueue.popleft()
        for child in childrenTree[item]:
            #Message to child: parent marginal divided by (message from child to parent), summed over all elements except sepset.
            #backwardsQueue.append(child)
            belief = marginalAtEachNode[item]
            varsInCurrentNode = mapping[item]
            varsInSeparator = factorMapping[separatorMemoized[child, item]]
            messageToChild = {x:0 for x in xrange(pow(10,len(varsInSeparator)))}

            tempMaxFinder = {x:[] for x in xrange(pow(10,len(varsInSeparator)))}

            for i in xrange(pow(10, len(varsInCurrentNode))):
                assignment = numberToAssignment(i, len(varsInCurrentNode))
                assignmentInMessage = [assignment[varsInCurrentNode.index(x)] for x in varsInSeparator]
                indexInMessage = assignmentToNumber(assignmentInMessage)
                tempMaxFinder[indexInMessage].append(belief[i])
                #messageToChild[indexInMessage] += exp(belief[i])

            for i in tempMaxFinder.keys():
                messageToChild[i] = max(tempMaxFinder[i])

            #Normalize message to child by factor division
            relevantSeparator = factorRepo[separatorMemoized[(child, item)]]
            #print relevantSeparator
            #print messageToChild
            messageToChild = {x:messageToChild[x]- relevantSeparator[x] for x in messageToChild.keys()}

            #Add the backwards message to the belief of the child
            updatedBelief = marginalAtEachNode[child]
            varsInChildMarginal = mapping[child]
            for i in updatedBelief.keys():
                assignment = numberToAssignment(i, len(varsInChildMarginal))
                assignmentInMessage = [assignment[varsInChildMarginal.index(x)] for x in varsInSeparator]
                updatedBelief[assignmentToNumber(assignment)] += messageToChild[assignmentToNumber(assignmentInMessage)]

            marginalAtEachNode[child] = updatedBelief

            backwardsQueue.append(child)



    end = time.clock()
    duration = end - start

    return marginalAtEachNode, mapping, duration

def getPartitionFunction(w1, w2, model):
    marginalAtCluster = forwardBackward(w1, w2, model)[0][0]
    #print marginalAtCluster

    commonFactor = min(marginalAtCluster.values())
    sum = 0
    for i in marginalAtCluster.keys():
        sum += exp(marginalAtCluster[i] - commonFactor)

    sum *= exp(commonFactor)
    return sum

def getMaxMarginalAssignment(w1, w2, model):
    marginals,mapping,duration = forwardBackward(w1, w2, model)

    #Variable to cluster assignment
    varMapping = {}

    for i in xrange(len(w1) + len(w2)):
        for j in mapping.keys():
            vars = mapping[j]
            if i in vars:
                varMapping[i] = j
                break

    finalMarginals = {x:{} for x in xrange(len(w1) + len(w2))}
    for i in varMapping.keys():
        varsInCluster = mapping[varMapping[i]]
        index = varsInCluster.index(i)
        finalMarginals[i] = {x:0 for x in characterArray}

        #tempArray = []

        for x in marginals[varMapping[i]].keys():
            assignment = numberToAssignment(x, len(varsInCluster))
            finalMarginals[i][assignment[index]] += exp(marginals[varMapping[i]][x])
            #tempArray.append(marginals[varMapping[i]][x])

        #commonFactor = min(tempArray)

        #tempArray = [exp(k - commonFactor) for k in tempArray]

        #finalMarginals[i][assignment[index]] = sum(tempArray)*exp(commonFactor)

    #print finalMarginals

    maxMarginalAssignment = {}

    for x in finalMarginals.keys():
        max = -500
        for y in finalMarginals[x].keys():
            if finalMarginals[x][y]>max:
                max = finalMarginals[x][y]
                maxTerm = y
        maxMarginalAssignment[x] = maxTerm

    return maxMarginalAssignment, duration

def getLogLikelihood(w1,w2,a1,a2,model):
    marginals,mapping, duration = forwardBackward(w1, w2, model)

    #Variable to cluster assignment
    varMapping = {}

    for i in xrange(len(w1) + len(w2)):
        for j in mapping.keys():
            vars = mapping[j]
            if i in vars:
                varMapping[i] = j
                break

    finalMarginals = {x:{} for x in xrange(len(w1) + len(w2))}

    for i in varMapping.keys():
        varsInCluster = mapping[varMapping[i]]
        index = varsInCluster.index(i)
        finalMarginals[i] = {x:0 for x in characterArray}

        for x in marginals[varMapping[i]].keys():
            assignment = numberToAssignment(x, len(varsInCluster))
            finalMarginals[i][assignment[index]] += exp(marginals[varMapping[i]][x])

    #print finalMarginals

    for k,v in finalMarginals.items():
        z = 0
        for val in v.values():
            z += val
        for x in v.keys():
            v[x] = v[x]/z
        #print sum(v.values())
        finalMarginals[k] = v
        #print sum(finalMarginals[k].values())

    #print finalMarginals

    LL = 0

    #print finalMarginals

    for i in xrange(len(w1)):
        LL += log(finalMarginals[i][a1[i]])

    for i in xrange(len(w2)):
        #print i, a2[i], i + len(w1)
        LL += log(finalMarginals[i + len(w1)][a2[i]])

    return LL, duration


def getMAPAssignment(w1, w2, model):
    marginal, mapping, duration = forwardBackward(w1,w2,model)

    varsLeft = {x for x in xrange(len(w1) + len(w2))}

    MAPatEachClique = {}

    for j in mapping.keys():
        varsInClique = mapping[j]

        maxProb = max(marginal[j].values())
        maxAss = -1

        for i in marginal[j].keys():
            if maxProb==marginal[j][i]:
                maxAss = i

        assignment = numberToAssignment(maxAss, len(varsInClique))

        for i in xrange(len(varsInClique)):
            MAPatEachClique[varsInClique[i]] = assignment[i]

    #print MAPatEachClique

    return  MAPatEachClique, duration







readOCRPotentials()
readTransPotentials()




#print assignmentToNumber(['e', 'o', 'i'])
#print numberToAssignment(430)
#forwardBackward([711,438,883,455,100,438],[392,438,266,584,455,600],['a','b','c','d','e','b'],['f','b','g','h','d','i'], 4)
#print forwardBackward([405, 840, 840], [826, 623, 840], ['a', 'b', 'b'], ['f', 'g', 'b'], 4)
#print getPartitionFunction([711,438,883,455,100,438],[392,438,266,584,455,600],['a','b','c','d','e','b'],['f','b','g','h','d','i'], 4)
#print assignmentToNumber(['n', 'd', 'd'])
#print getMaxMarginalAssignment([405, 840, 840], [826, 623, 840], 4)
#print forwardBackward([542,949,830], [742,981,543,625,830,758], 4)
#print getLogLikelihood([192,551,450], [597,192,75], ['a','e','d'], ['e', 't', 'o'], 4)

#print getPartitionFunction([542,949,830], [742,981,543,625,830,758], 1)
#print getPartitionFunction([542,949,830], [742,981,543,625,830,758], 2)
#print getPartitionFunction([542,949,830], [742,981,543,625,830,758], 3)
#print getPartitionFunction([542,949,830], [742,981,543,625,830,758], 4)
#print getPartitionFunction([982,734,608,681], [999,80,678,982,521], 4)


getMAPAssignment([405, 840, 840], [826, 623, 840], 4)
print '\n\n\n'
getMAPAssignment([711,438,883,455,100,438],[392,438,266,584,455,600], 4)