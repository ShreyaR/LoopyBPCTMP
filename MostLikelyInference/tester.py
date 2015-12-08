__author__ = 'shreyarajpal'

from math import exp, log
'''
from helperFunctionsRepo import ocrDat, transDat, assignmentToNumber, numberToAssignment, onlyTemporary

print pow(10,2)

termsInCurrentNode = [3,4]
termsInMessage = [4]
w1 = [405, 840, 840]
w2 = [826, 623, 820]

psiFactor = {}

for i in xrange(pow(10, 2)):
    a = numberToAssignment(i, 2)
    potential = 0
    f1 = 826
    f2 = 623

    potential += ocrDat[f1][a[0]]
    potential += ocrDat[f2][a[1]]
    potential += transDat[a[0]][a[1]]

    psiFactor[i] = potential

storage = []
storage.append(psiFactor)

tauStore = []

tauFactor = {}
for i in xrange(pow(10,1)):
    a = numberToAssignment(i,1)
    potential = 1

    for j in xrange(pow(10,1)):
        b = numberToAssignment(j,1)
        x = [b[0], a[0]]
        potential += exp(psiFactor[assignmentToNumber(x)])

    tauFactor[assignmentToNumber(a[0])] = log(potential)


tauStore.append(tauFactor)

#print tauFactor
#print psiFactor

psiUpdate = {}
for i in xrange(pow(10,2)):
    potential = 0
    a = numberToAssignment(i,2)
    potential += transDat[a[0]][a[1]]
    potential += tauStore[0][assignmentToNumber(a[0])]

    psiUpdate[i] = potential

#print psiUpdate
storage.append(psiUpdate)

for i in xrange(pow(10,1)):
    a = numberToAssignment(i,1)
    potential = 1

    for j in xrange(pow(10,1)):
        b = numberToAssignment(j,1)
        x = [b[0], a[0]]
        potential += exp(psiUpdate[assignmentToNumber(x)])

    tauFactor[assignmentToNumber(a[0])] = log(potential)


tauStore.append(tauFactor)


psiUpUpdate = {}
for i in xrange(pow(10,1)):
    potential = 0
    a = numberToAssignment(i,1)
    potential += tauStore[1][assignmentToNumber(a[0])]

    psiUpUpdate[i] = potential

#print psiUpUpdate

storage.append(psiUpUpdate)

tauFactor = {}




for i in xrange(pow(10,1)):
    a = numberToAssignment(i,1)
    potential = 0
    #print j
    b = numberToAssignment(j,1)
    x = [a[0]]
    potential += exp(psiUpUpdate[assignmentToNumber(x)])
    #print potential

    tauFactor[assignmentToNumber(a[0])] = log(potential)

#print 'tauFactor is',tauFactor

psiUpUpUpdate = {}

for i in xrange(pow(10,3)):
    #Arrangement like 1,2,5
    a = numberToAssignment(i,3)
    potential = 0
    potential += ocrDat[840][a[2]]
    potential += ocrDat[840][a[1]]
    if a[1] == a[2]:
        potential += log(5)

    if a[1] == a[0]:
        potential += log(5)

    if a[0] == a[2]:
        potential += log(5)

    potential += transDat[a[0]][a[1]]
    potential += tauFactor[assignmentToNumber(a[2])]

    psiUpUpUpdate[i] = potential



tauStore.append(tauFactor)


#print psiUpUpUpdate


for i in xrange(1000):
    print onlyTemporary[i], psiUpUpUpdate[i], psiUpUpUpdate[i]==onlyTemporary[i]

tauFactor = {}
for i in xrange(10):
    a = numberToAssignment(i,1)
    potential = 1
    for i in xrange(100):
        newAss = numberToAssignment(i,2)
        b = [a[0],newAss[0], newAss[1]]
        potential += exp(psiUpUpUpdate[assignmentToNumber(b)])

    tauFactor[assignmentToNumber(a[0])] = log(potential)

print tauFactor


psiUpUpUpUpdate = {}
for i in xrange(100):
    #In order [0,1]
    assignment = numberToAssignment(i,2)

    potential = 0
    potential = potential + ocrDat[405][assignment[0]] + ocrDat[840][assignment[1]] + transDat[assignment[0]][assignment[1]]
    potential += tauFactor[assignmentToNumber([assignment[1]])]

    psiUpUpUpUpdate[i] = potential

print psiUpUpUpUpdate

'''




from loopyBP import transDat, numberToAssignment, assignmentToNumber, marginalize, updateBelief
beliefAt6 = {}
marginalizedBelief = {x:0 for x in xrange(10)}

for i in xrange(100):
    assignment = numberToAssignment(i,2)
    beliefAt6[i] = transDat[assignment[0]][assignment[1]]
    assToSep = assignment[0]
    marginalizedBelief[assignmentToNumber([assToSep])] += exp(beliefAt6[i])

marginalizedBelief = {x:log(marginalizedBelief[x]) for x in marginalizedBelief.keys()}

print beliefAt6

mergB = marginalize(beliefAt6, 0, (0,1), 't')

for i in mergB.keys():
    print marginalizedBelief[i], mergB[i], mergB[i]==marginalizedBelief[i]

'''
updatedBelief = {}
for i in xrange(100):
    assignment = numberToAssignment(i,2)
    assToSep = assignmentToNumber(assignment[1])

    updatedBelief[i] = beliefAt6[i] + marginalizedBelief[assToSep]


edgeB = {x:0 for x in xrange(10)}

upB = updateBelief(beliefAt6, marginalizedBelief, 1, (0,1), edgeB, 't')

for i in upB.keys():
    print updatedBelief[i], upB[i], upB[i]==updatedBelief[i]

'''
