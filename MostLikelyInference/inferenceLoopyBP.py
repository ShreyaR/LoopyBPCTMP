__author__ = 'shreyarajpal'

import time
global loopWords, loopWordsWS, treeWords, treeWordsWS
#from helperFunctions import ocrDat, transDat, getMaxMarginalAssignment, getLogLikelihood

from loopyBP import getLogLikelihood, getMAPassignment

loopWords = 'data/data-loops.dat'
loopWordsWS = 'data/data-loopsWS.dat'
treeWords = 'data/data-tree.dat'
treeWordsWS = 'data/data-treeWS.dat'

trueLoopWords = 'data/truth-loops.dat'
trueLoopWordsWS = 'data/truth-loopsWS.dat'
trueTreeWords = 'data/truth-tree.dat'
trueTreeWordsWS = 'data/truth-treeWS.dat'

def infer(data):
    if data==1:
        dataFile = open(loopWords)
        trueFile = open(trueLoopWords)
    elif data==2:
        dataFile = open(loopWordsWS)
        trueFile = open(trueLoopWordsWS)
    elif data==3:
        dataFile = open(treeWords)
        trueFile = open(trueTreeWords)
    else:
        dataFile = open(treeWordsWS)
        trueFile = open(trueTreeWordsWS)

    words = []
    assignments = []
    while(True):
        chunk = dataFile.read().split('\n')
        for i in xrange(len(chunk)/3):
            w1,w2 = chunk[i*3],chunk[i*3 + 1]
            w1 = [int(i) for i in w1.rstrip().split('\t')]
            w2 = [int(i) for i in w2.rstrip().split('\t')]
            words.append((w1,w2))
        dataFile.close()
        break

    while(True):
        chunk = trueFile.read().split('\n')
        for i in xrange(len(chunk)/3):
            w1,w2 = chunk[i*3],chunk[i*3 + 1]
            assignments.append((w1,w2))
        trueFile.close()
        break

    inference = []
    totalTime = 0

    for pair in words:
        w1,w2 = pair
        print 1

        inference1, inference2, duration = getMAPassignment(w1, w2, 4)
        print 2
        totalTime += duration
        inference.append((inference1,inference2))

    #print inference

    totalWords = 2*len(assignments)
    totalCharacters = 0

    correctWords = 0
    correctCharacters = 0


    LL = 0

    for i in xrange(len(assignments)):

        predictedWord1, predictedWord2 = inference[i]
        actualWord1, actualWord2 = assignments[i]
        image1, image2 = words[i]

        if predictedWord1==actualWord1:
            correctWords += 1

        if predictedWord2==actualWord2:
            correctWords += 1

        totalCharacters += len(actualWord1) + len(actualWord2)

        for x in xrange(len(predictedWord1)):
            if predictedWord1[x]==actualWord1[x]:
                correctCharacters += 1

        for x in xrange(len(predictedWord2)):
            if predictedWord2[x]==actualWord2[x]:
                correctCharacters += 1

        #t1, duration = getLogLikelihood(image1, image2, actualWord1, actualWord2, 4)
        #LL += t1
        totalTime += duration



    #print LL
    #print totalWords

    print 'charAccuracy = ',float(correctCharacters)/totalCharacters
    print 'wordAccuracy = ', float(correctWords)/totalWords
    print 'logLikelihood = ', LL/totalWords
    print 'totalTime = ', totalTime
    return


infer(3)
infer(4)
infer(1)
infer(2)