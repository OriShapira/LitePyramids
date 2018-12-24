import sys
import spacy
import csv
import random
import nltk

'''
The script is part of the first phase of the LitePyramids.

It creates a CSV file containing the SCUs to use for the next phase (system summary testing)
based on the statements generated by crowdworkers on AMT.

Provide the input and output files below in the variables.
'''

'''
The input file to this script is the AMT downloaded result CSV file for the SCU building task.
These are the fields of the result file:
    "HITId"
    "HITTypeId"
    "Title"
    "Description"
    "Keywords"
    "Reward"
    "CreationTime"
    "MaxAssignments"
    "RequesterAnnotation"
    "AssignmentDurationInSeconds"
    "AutoApprovalDelayInSeconds"
    "Expiration"
    "NumberOfSimilarHITs"
    "LifetimeInSeconds"
    "AssignmentId"
    "WorkerId"
    "AssignmentStatus"
    "AcceptTime"
    "SubmitTime"
    "AutoApprovalTime"
    "ApprovalTime"
    "RejectionTime"
    "RequesterFeedback"
    "WorkTimeInSeconds"
    "LifetimeApprovalRate"
    "Last30DaysApprovalRate"
    "Last7DaysApprovalRate"
    "Input.eventId"
    "Input.summId"
    "Input.summary_text"
    "Answer.s1"
    "Answer.s2"
    "Answer.s3"
    "Answer.s4"
    "Answer.s5"
    "Answer.s6"
    "Answer.s7"
    "Answer.s8"
    "Approve"
    "Reject"
'''
SCUS_RESULTS_CSV = '' # e.g. 'fromAMT/Batch_1234567_batch_results.csv'
'''
The CSV file to output the final questions file.
This is used for creating the task of the next phase (system summary testing).
The fields are:
    eventId,questionId,questionText,answer,author,sourceSummaryId,forUse
'''
OUTPUT_CSV_PATH = '' # e.g. 'SCUs_batch1.csv'
'''
The max number of SCUs to use for each reference summary when testing in the next phase (these will be marked in the dataset).
(There may be less available.)
'''
NUM_SCUS_TO_SAMPLE = 4
'''
The similarity score to use for finding similar statements generated for the same reference summary.
This is on a lemmatized, stop-words-removed, bag-of-words ratio based similarity.
I.e. percentage of the lemmas that overlap.
'''
SIMILARITY_SCORE_BAGOFWORDS = 0.95




nlp = spacy.load('en_core_web_sm')

def main(scusCsvFile, outputCsvFile):
    # get the SCUs from the mechanical turk file:
    allScus, allScusAuthors = getAllScus(scusCsvFile)
    # get 8 SCUs and 4 sampled SCUs for each event from the full list of SCUs written by turkers:
    chosenScusIndices = {} # the 32 SCUs chosen for the event (8 per reference summary)
    finalScusIndices = {} # the 16 SCUs to be used for the evaluation process (4 sampled per reference summary)
    for eventId in allScus:
        chosenScusIndicesForEvent, finalScusIndicesForEvent = getFinalScus(allScus, eventId)
        chosenScusIndices[eventId] = chosenScusIndicesForEvent
        finalScusIndices[eventId] = finalScusIndicesForEvent
        
    # output the final SCUs to a quetions CSV to be used in the evaluation database:
    outputQuestionsCSV(outputCsvFile, allScus, allScusAuthors, chosenScusIndices, finalScusIndices)

def getAllScus(scusCsvFile):
    # Get the statments generated by crowdworkers from the AMT batch file.
    # Returns:
    #   dictionary of { eventId -> { summId -> [s1,...,s8] } }
    #   dictionary of { eventId -> { summId -> [workerId,...,workerId] } }
    
    allScus = {}
    allScusAuthors = {}
    with open(scusCsvFile, 'r') as inF:
        csv_reader = csv.DictReader(inF)
        for row in csv_reader:
            eventId = row['Input.eventId']
            summId = row['Input.summId']
            workerId = row['WorkerId']
            s1, s2, s3, s4, s5, s6, s7, s8 = \
                row['Answer.s1'], \
                row['Answer.s2'], \
                row['Answer.s3'], \
                row['Answer.s4'], \
                row['Answer.s5'], \
                row['Answer.s6'], \
                row['Answer.s7'], \
                row['Answer.s8']
                
            allScus.setdefault(eventId, {}).setdefault(summId, []).extend([s1, s2, s3, s4, s5, s6, s7, s8])
            allScusAuthors.setdefault(eventId, {}).setdefault(summId, []).extend([workerId]*8)
    
    return allScus, allScusAuthors

def getFinalScus(allScus, eventId):
    # get the SCUs from allSCUs to be used in the questions CSV for the evaluation database

    chosenScusIndices = {} # the indices of the SCUs in allScus[eventId] that will be sampled from
    finalScusIndices = {} # the indices of the SCUs in allScus[eventId] to use for system summary evaluation
    
    for summId in allScus[eventId]:
        chosenScusIndices[summId] = []
        finalScusIndices[summId] = []
    
        # get the relevant SCUs for the current reference summary:
        scusForSummaryIndices = getScusForSummary(allScus[eventId][summId])
        # choose 4 random SCUs for the current reference summary to use for system summary evaluation:
        sampledScusForSummaryIndices = getSampleFromList(scusForSummaryIndices, NUM_SCUS_TO_SAMPLE)
        
        chosenScusIndices[summId].extend(scusForSummaryIndices)
        finalScusIndices[summId].extend(sampledScusForSummaryIndices)
    
    return chosenScusIndices, finalScusIndices
    
def getSampleFromList(listToSampleFrom, sampleSize):
    return [listToSampleFrom[i] for i in sorted(random.sample(xrange(len(listToSampleFrom)), sampleSize))]
    
def outputQuestionsCSV(outputCsvPath, allScus, allScusAuthors, chosenScusIndices, finalScusIndices):
    # Write out the final CSV for the SCUs. Those to be used in the system summary evalaution phase are marked.
    with open(outputCsvPath, 'w') as outF:
        qId = 0
        outF.write('eventId,questionId,questionText,answer,author,sourceSummaryId,forUse\n')
        for eventId in allScus:
            for summId in allScus[eventId]:
                for ind in chosenScusIndices[eventId][summId]:
                    outF.write('{},{},"{}",{},{},{},'.format(eventId, qId, allScus[eventId][summId][ind], 'Y', allScusAuthors[eventId][summId][ind], summId))
                    if ind in finalScusIndices[eventId][summId]:
                        outF.write('1\n')
                        print(eventId, allScusAuthors[eventId][summId][ind], summId, allScus[eventId][summId][ind])
                    else:
                        outF.write('0\n')
                    qId += 1
    


def getScusForSummary(scusList):
    # initially, use all SCUs, and from here start removing irrelevant ones:
    chosenIndices = range(len(scusList))
    # create SpaCy objects for all the SCUs (strip basic punctuation):
    scuDocs = [nlp(unicode(scu.strip('.,!?'))) for scu in scusList]
    # create SpaCy objects for all the SCUs as tokens and without stop words:
    scuDocsBase = [nlp(' '.join([token.lemma_ for token in scuDoc if not token.is_stop])) for scuDoc in scuDocs]
    
    # remove SCUs with more than one sentence, or with more than 20 words, or with less than 4:
    for scuIdx, scuDoc in enumerate(scuDocs):
        # if there's more than one sentence, don't use it:
        if len(list(scuDoc.sents)) > 1:
            chosenIndices.remove(scuIdx)
            #print('>1 sent: ' + str(scuDoc))
        # if the sentence is longer than 20 words, don't use it:
        elif len(scuDoc) > 20:
            chosenIndices.remove(scuIdx)
            #print('Too long: ' + str(scuDoc))
        elif len(scuDoc) <= 3:
            chosenIndices.remove(scuIdx)
            print('Too short: ' + str(scuDoc))
        
    # compute the similarities between all tokenized SCU pairs, and remove SCUs that are the similar previous ones:
    removeList = []
    for scuIdx1 in chosenIndices:
        if scuIdx1 not in removeList:
            scuDoc1 = scuDocsBase[scuIdx1]
            for scuIdx2 in chosenIndices:
                if scuIdx1 < scuIdx2 and scuIdx2 not in removeList:
                    scuDoc2 = scuDocsBase[scuIdx2]
                    if isSimilarOverlap(scuDoc1, scuDoc2):
                        # if the scus are similar, remove the longer one:
                        if len(scuDoc1) < len(scuDoc2) and scuIdx2 not in removeList:
                            removeList.append(scuIdx2)
                        elif scuIdx1 not in removeList:
                            removeList.append(scuIdx1)
                        #print('-', str(scuDocs[scuIdx1]), str(scuDocs[scuIdx2]))
                        print(str(scuDocs[scuIdx1]), str(scuDocs[scuIdx2]))
                    
    # filter out the ones to remove:
    for scuIdxToRemove in removeList:
        chosenIndices.remove(scuIdxToRemove)
        
    # Until now chosenIndices are all SCUs with one sentence, less than 20 tokens, more than 3 tokens, and with no repetitions of quite similar SCUs.
    
    return chosenIndices

def isSimilarW2V(scuDoc1, scuDoc2):
    retVal = False
    sim = scuDoc1.similarity(scuDoc2)
    if sim > 0.9:
        retVal = True
    return retVal
    
def isSimilarOverlap(scuDoc1, scuDoc2):
    retVal = False
    set1 = set([n for n in nltk.ngrams([str(token) for token in scuDoc1], 1)])
    set2 = set([n for n in nltk.ngrams([str(token) for token in scuDoc2], 1)])
    intersectionSize = float(len(set1 & set2))
    intersectionRatio = intersectionSize / min(len(set1), len(set2))
    if intersectionRatio > SIMILARITY_SCORE_BAGOFWORDS:
        retVal = True
    return retVal
    
if __name__ == '__main__':
    main(SCUS_RESULTS_CSV, OUTPUT_CSV_PATH)