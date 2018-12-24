import sys
import csv
import ast
from collections import Counter
import copy
import difflib
from scipy.stats import pearsonr
from scipy.stats import spearmanr
import random
import time
import operator
import numpy as np

'''
This script get scores and correlations of systems according to the Lite-Pyramid evaluation method, based on
crowdsourced SCU judgments.
Run: python post_calculateScores.py [-scores|-corr]
    -scores outputs only the scores to the output file
    -corr   also outputs the correlation of the scores to original Pyramid, as well as Responsiveness and ROUGE to Pyramid
    default is scores
    
Provide the input and output files, and the configuration in the variables below.

Since this task is run twice (one for each 16-SCU batch), combine the two AMT downloaded results files, and run
this script on the combined AMT file (don't copy the header line from one file to the other).
'''


'''
The input file to this script is the AMT downloaded result CSV file for the system summary evaluation task.
These are the fields of the result file expected:
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
    "Input.summaryId"
    "Input.qIdList"
    "Input.summary_text"
    "Input.statement_1"
    "Input.statement_2"
    "Input.statement_3"
    "Input.statement_4"
    "Input.statement_5"
    "Input.statement_6"
    "Input.statement_7"
    "Input.statement_8"
    "Input.statement_9"
    "Input.statement_10"
    "Input.statement_11"
    "Input.statement_12"
    "Input.statement_13"
    "Input.statement_14"
    "Input.statement_15"
    "Input.statement_16"
    "Answer.S10Answer"
    "Answer.S11Answer"
    "Answer.S12Answer"
    "Answer.S13Answer"
    "Answer.S14Answer"
    "Answer.S15Answer"
    "Answer.S16Answer"
    "Answer.S1Answer"
    "Answer.S2Answer"
    "Answer.S3Answer"
    "Answer.S4Answer"
    "Answer.S5Answer"
    "Answer.S6Answer"
    "Answer.S7Answer"
    "Answer.S8Answer"
    "Answer.S9Answer"
    "Approve"
    "Reject"
'''
RESULTS_FILE_INPUT = '' # e.g. 'fromAMT/Batch_7654321_batch_results.csv'
'''
The CSV file to output the results to.
This is used for creating the task of the next phase (system summary testing).
When getting correlations, the fields are:
    ANSWER_AGGREGATION_TYPE
    ANSWER_TIE_BREAKER
    NO_ANSWER_DEFAULT
    NUM_TURKERS_PER_SUMMARY
    NUM_QUESTIONS_PER_SUMMARY
    NUM_EVENTS_TO_USE
    WORKER_AGREEMENT_THRESHOLD
    AGREEMENT_FILTERING_ITERATIONS
    pearsonCorr
    pearsonPVal
    spearmanCorr
    spearmanPVal
    rankOrig
    rankOurs
    pCorrResp
    pPvalResp
    sCorrResp
    sPvalResp
    pCorrR1
    pPvalR1
    sCorrR1
    sPvalR1
    pCorrR2
    pPvalR2
    sCorrR2
    sPvalR2
    pCorrRL
    pPvalRL
    sCorrRL
    sPvalRL
When getting just scores, the fields are:
    ANSWER_AGGREGATION_TYPE
    ANSWER_TIE_BREAKER
    NO_ANSWER_DEFAULT
    NUM_TURKERS_PER_SUMMARY
    NUM_QUESTIONS_PER_SUMMARY
    NUM_EVENTS_TO_USE
    WORKER_AGREEMENT_THRESHOLD
    AGREEMENT_FILTERING_ITERATIONS
    rankOrig
    rankOurs
'''
OUTPUT_FILE = '' # e.g. 'results.csv'
'''
The path to the CSV file with the original manual scores as created by the getManualScores.py script.
'''
MANUAL_SCORES_FILE = '' # e.g. '../score_extraction/2006ManualScoresAvg.csv'
'''
The path to the CSV file with the original automatic scores as created by the getRougeScores.py script.
'''
ROUGE_SCORES_FILE = '' # e.g. '../score_extraction/2006RougeScoresAvg.csv'


### Configuration options:
# How to decide on an answer when several are provided
ANSWER_AGGREGATION_TYPE = [1] #[0, 1, 2] # 0=average 1=majority 2=atleast1present
# If there happens to be an even number of answers and there's a tie, what is the answer:
ANSWER_TIE_BREAKER = [0.0] # 0.0=NotPresent, 1.0=Present, 0.5=Gives half a point
# If there's no given answer by a turker, what value should it be assigned:
NO_ANSWER_DEFAULT = [1.0] # 0.0=NotPresent, 1.0=Present, 0.5=Ignore
# The number of turkers (assignments) to use per summary:
NUM_TURKERS_PER_SUMMARY = [5] #[1, 3, 5] # if more than this number, takes random sample, if less, takes all
# The number of questions to use for each summary evaluation
NUM_QUESTIONS_PER_SUMMARY = [32]#[16, 24, 32] # if more than this number, takes random sample (consistent over event), if less, takes all
# The number of events on which to evaluate:
NUM_EVENTS_TO_USE = [20] # [16,18,20] # if more than this number, takes random sample, if less, takes all
# The agreement threshold below which to disregard a workers answers:
WORKER_AGREEMENT_THRESHOLD = [0.5] # if the agreement of a worker is below this threshold, then the worker is filtered out
## TODO: take into account how many assignments the worker has done
# How many times should agreement filtering be done (after removing some, filter again on the remaining):
AGREEMENT_FILTERING_ITERATIONS = [1] #[1, 2, 3]
# What percent of the events to filter by agreement of event answers:
EVENT_FILTER_PERCENT = [0.0] # [0.0, 0.2]
# How many times should each configuration be tested and then averaged:
NUM_ITERATION_ON_CONFIGURATION = 70


def computeScoresAndCorrelations(rawDataValues, questionIdsPerEvent, systemScoresAllOrig, configuration, onlyScores=True):
    # correlations between pyramid and our method (lists over iterations)
    pearsonCorrsAll = [] # [pCorr to pyr] for our method
    pearsonPValuesAll = [] # [pPval to pyr] for our method
    spearmanCorrsAll = [] # [sCorr to pyr] for our method
    spearmanPValuesAll = [] # [sPval to pyr] for our method
    # the actual scores using our method (lists over events used)
    systemScoresOursAll = {} # { systemId -> [our scores over events] }
    systemScoresOriginalAll = {} # { systemId -> [pyramid scores over events] }
    
    # correlations between pyramid and other original methods (lists over iterations):
    pearsonCorrsOrigAll = {'resp':[],'r1':[],'r2':[],'rL':[]} # { <'resp'/'r1'/'r2'/'rL'> -> [pCorr to pyr] }
    pearsonPValuesOrigAll = {'resp':[],'r1':[],'r2':[],'rL':[]} # { <'resp'/'r1'/'r2'/'rL'> -> [pPval to pyr] }
    spearmanCorrsOrigAll = {'resp':[],'r1':[],'r2':[],'rL':[]} # { <'resp'/'r1'/'r2'/'rL'> -> [sCorr to pyr] }
    spearmanPValuesOrigAll = {'resp':[],'r1':[],'r2':[],'rL':[]} # { <'resp'/'r1'/'r2'/'rL'> -> [sPval to pyr] }
    
    
    # get the data in a mapped-value format (dependent on some configuration parameters):
    dataValues = mapValues(
        rawDataValues,
        configuration['NO_ANSWER_DEFAULT'])
    
    # get a list of workers to disregard during scoring:
    workersToFilter = getWorkersToFilter(
        dataValues,
        questionIdsPerEvent,
        configuration['WORKER_AGREEMENT_THRESHOLD'],
        configuration['AGREEMENT_FILTERING_ITERATIONS'])
        
    # get the event agreement scores list in case needed:
    eventAgreements = _measureEventAgreement(rawDataValues, workersToFilter)
    
    
    # run several iterations on the current configuration to get an average (since there's randomization):
    for i in range(configuration['NUM_ITERATION_ON_CONFIGURATION']):
        
        # get a list of events to disregard during scoring:
        #eventsToFilter = getEventsToFilter(
        #    rawDataValues,
        #    workersToFilter,
        #    configuration['EVENT_FILTER_PERCENT'],
        #    printToScreen=False)
        #eventsToFilter = _filterEventsByAgreement(eventAgreements, )
        
        # get the scores (in our method) of each system summary:
        summaryScores = getSystemSummaryScores(
            dataValues,
            workersToFilter,
            questionIdsPerEvent,
            configuration['ANSWER_AGGREGATION_TYPE'],
            configuration['ANSWER_TIE_BREAKER'],
            configuration['NUM_QUESTIONS_PER_SUMMARY'],
            configuration['NUM_TURKERS_PER_SUMMARY'],
            configuration['NUM_EVENTS_TO_USE'],
            eventAgreements,
            configuration['EVENT_FILTER_PERCENT'])
            
        # for debugging - print the scores per topic:
        #printAverageEventScores(summaryScores)
        
        # get the system scores (in our method) according to their summary scores:
        systemScoresOurs, eventIdsUsedPerSystem = getSystemScores(summaryScores)
        for systemId in systemScoresOurs:
            systemScoresOursAll.setdefault(systemId, []).append(systemScoresOurs[systemId])
        
        # get the systems' original pyramid scores:
        systemScoresOrig = getOriginalScores(systemScoresAllOrig, eventIdsUsedPerSystem)
        for systemId in systemScoresOrig['pyr']:
            systemScoresOriginalAll.setdefault(systemId, []).append(systemScoresOrig['pyr'][systemId])
        
        # if we also need to get correlations:
        if not onlyScores:
        
            # correlate the system scores of our method, to those in the pyramids:
            pearsonCorrOurs, pearsonPValueOurs, spearmanCorrOurs, spearmanPValueOurs = \
                getSystemScoreCorrelations(systemScoresOrig['pyr'], systemScoresOurs)
            # also get the correlations between the pyramid method and the other original evaluation methods:
            correlationsBetweenOriginals = getCorrelationsBetweenOriginalScores(systemScoresOrig)
            
            # append our correlations for the current iteration:
            pearsonCorrsAll.append(pearsonCorrOurs)
            pearsonPValuesAll.append(pearsonPValueOurs)
            spearmanCorrsAll.append(spearmanCorrOurs)
            spearmanPValuesAll.append(spearmanPValueOurs)
            
            # append original method correlations for the current iteration:
            for method in ['resp', 'r1', 'r2', 'rL']:
                pearsonCorrsOrigAll[method].append(correlationsBetweenOriginals[method]['pV'])
                pearsonPValuesOrigAll[method].append(correlationsBetweenOriginals[method]['pP'])
                spearmanCorrsOrigAll[method].append(correlationsBetweenOriginals[method]['sV'])
                spearmanPValuesOrigAll[method].append(correlationsBetweenOriginals[method]['sP'])
                
            #print(pearsonCorrOurs, pearsonPValueOurs, spearmanCorrOurs, spearmanPValueOurs)
    
    
    # now that we've finished running many iterations, calculate the average system scores over the iterations:
    systemScoresOursFinal = {systemId:reduce(lambda x, y: x + y, scoresList) / len(scoresList) 
        for systemId, scoresList in systemScoresOursAll.items()}
    systemScoresOriginalFinal = {systemId:reduce(lambda x, y: x + y, scoresList) / len(scoresList) 
        for systemId, scoresList in systemScoresOriginalAll.items()}
    
    # if we also need to get correlations:
    if not onlyScores:
        # now that we've finished running many iterations, calculate the average correlations of our method over all iterations:
        pearsonCorrFinal = np.mean(pearsonCorrsAll) # reduce(lambda x, y: x + y, pearsonCorrsAll) / len(pearsonCorrsAll)
        pearsonCorrFinalStd = np.std(pearsonCorrsAll)
        pearsonPValueFinal = np.mean(pearsonPValuesAll) # reduce(lambda x, y: x + y, pearsonPValuesAll) / len(pearsonPValuesAll)
        spearmanCorrFinal = np.mean(spearmanCorrsAll) # reduce(lambda x, y: x + y, spearmanCorrsAll) / len(spearmanCorrsAll)
        spearmanCorrFinalStd = np.std(spearmanCorrsAll)
        spearmanPValueFinal = np.mean(spearmanPValuesAll) # reduce(lambda x, y: x + y, spearmanPValuesAll) / len(spearmanPValuesAll)
    
        # calculate the average correlations of original methods over the iterations:
        pearsonCorrOrigFinal = {}
        pearsonPValueOrigFinal = {}
        spearmanCorrOrigFinal = {}
        spearmanPValueOrigFinal = {}
        for method in ['resp', 'r1', 'r2', 'rL']:
            pearsonCorrOrigFinal[method] = reduce(lambda x, y: x + y, pearsonCorrsOrigAll[method]) / len(pearsonCorrsOrigAll[method])
            pearsonPValueOrigFinal[method] = reduce(lambda x, y: x + y, pearsonPValuesOrigAll[method]) / len(pearsonPValuesOrigAll[method])
            spearmanCorrOrigFinal[method] = reduce(lambda x, y: x + y, spearmanCorrsOrigAll[method]) / len(spearmanCorrsOrigAll[method])
            spearmanPValueOrigFinal[method] = reduce(lambda x, y: x + y, spearmanPValuesOrigAll[method]) / len(spearmanPValuesOrigAll[method])
    
        return pearsonCorrFinal, pearsonCorrFinalStd, pearsonPValueFinal, spearmanCorrFinal, spearmanCorrFinalStd, spearmanPValueFinal, \
            systemScoresOursFinal, systemScoresOriginalFinal, \
            pearsonCorrOrigFinal, pearsonPValueOrigFinal, spearmanCorrOrigFinal, spearmanPValueOrigFinal
            
    else:
        return None, None, None, None, None, None, systemScoresOursFinal, systemScoresOriginalFinal, None, None, None, None
    
def getCorrelationsBetweenOriginalScores(systemScoresOriginal):
    # get the correlations between the different original evaluation methods to pyramids:
    
    # Pyramid to Responsiveness
    pearsonCorrPyrResp, pearsonPValPyrResp, spearmanCorrPyrResp, spearmanPValPyrResp = \
        getSystemScoreCorrelations(systemScoresOriginal['pyr'], systemScoresOriginal['resp'])
    # Pyramid to ROUGE-1
    pearsonCorrPyrR1, pearsonPValPyrR1, spearmanCorrPyrR1, spearmanPValPyrR1 = \
        getSystemScoreCorrelations(systemScoresOriginal['pyr'], systemScoresOriginal['r1'])
    # Pyramid to ROUGE-2
    pearsonCorrPyrR2, pearsonPValPyrR2, spearmanCorrPyrR2, spearmanPValPyrR2 = \
        getSystemScoreCorrelations(systemScoresOriginal['pyr'], systemScoresOriginal['r2'])
    # Pyramid to ROUGE-L
    pearsonCorrPyrRL, pearsonPValPyrRL, spearmanCorrPyrRL, spearmanPValPyrRL = \
        getSystemScoreCorrelations(systemScoresOriginal['pyr'], systemScoresOriginal['rL'])
    
    correlations = {}    
    correlations['resp'] = {'pV': pearsonCorrPyrResp, 'pP': pearsonPValPyrResp, 'sV': spearmanCorrPyrResp, 'sP': spearmanPValPyrResp}
    correlations['r1'] = {'pV': pearsonCorrPyrR1, 'pP': pearsonPValPyrR1, 'sV': spearmanCorrPyrR1, 'sP': spearmanPValPyrR1}
    correlations['r2'] = {'pV': pearsonCorrPyrR2, 'pP': pearsonPValPyrR2, 'sV': spearmanCorrPyrR2, 'sP': spearmanPValPyrR2}
    correlations['rL'] = {'pV': pearsonCorrPyrRL, 'pP': pearsonPValPyrRL, 'sV': spearmanCorrPyrRL, 'sP': spearmanPValPyrRL}
    
    return correlations
    

def getSystemScoreCorrelations(systemScoresOriginal, systemScoresNew):
    # get the Pearson correlation and p-value, and the Spearman correlation and p-value between the two given scores dictionaries
    
    scoresNew = [systemScoresNew[systemId] for systemId in systemScoresNew]
    scoresOrig = [systemScoresOriginal[systemId] for systemId in systemScoresNew]
        
    pearsonCorr, pearsonPValue = pearsonr(scoresOrig, scoresNew)
    spearmanCorr, spearmanPValue = spearmanr(scoresOrig, scoresNew)
    
    return pearsonCorr, pearsonPValue, spearmanCorr, spearmanPValue
    
    
    
def readOriginalScoresData(originalManualScoresFile, originalRougeScoresFile):
    # read the original scores from the scores files as output by the getManualScores.py and getRougeScores.py scripts
    
    systemScoresAll = {'pyr':{}, 'resp':{}, 'r1':{}, 'r2':{}, 'rL':{}} # { <'pyr'/'resp'/'r1'/'r2'/'rL'> -> { systemId -> { eventId -> manualScore } } }
    
    with open(originalManualScoresFile, 'r') as inF:
        readValues = False
        for line in inF:
            line = line.strip()
            # start reading the data after this line (headers of the score data):
            if line == 'systemId, eventId, pyramid, responsiveness':
                readValues = True
            elif readValues == True:
                # split the comma delimited lines and get the values:
                values = [val.strip() for val in line.split(',')]
                if len(values) != 4:
                    continue
                systemId = values[0]
                eventId = values[1]
                pyramidScore = float(values[2])
                responsivenessScore = float(values[3])
                
                systemScoresAll['pyr'].setdefault(systemId, {})[eventId] = pyramidScore
                systemScoresAll['resp'].setdefault(systemId, {})[eventId] = responsivenessScore
                
    with open(originalRougeScoresFile, 'r') as inF:
        readValues = False
        for line in inF:
            line = line.strip()
            # start reading the data after this line (headers of the score data):
            if line == 'systemId, eventId , ROUGE-1 recall, ROUGE-1 precision, ROUGE-1 f1, ROUGE-2 recall, ROUGE-2 precision, ROUGE-2 f1, ROUGE-3 recall, ROUGE-3 precision, ROUGE-3 f1, ROUGE-4 recall, ROUGE-4 precision, ROUGE-4 f1, ROUGE-L recall, ROUGE-L precision, ROUGE-L f1, ROUGE-W-1.2 recall, ROUGE-W-1.2 precision, ROUGE-W-1.2 f1, ROUGE-SU4 recall, ROUGE-SU4 precision, ROUGE-SU4 f1':
                readValues = True
            elif readValues == True:
                # split the comma delimited lines and get the values:
                values = [val.strip() for val in line.split(',')]
                if len(values) != 23:
                    continue
                systemId = values[0]
                eventId = values[1]
                try:
                    rouge1Score = float(values[2])
                    rouge2Score = float(values[5])
                    rougeLScore = float(values[14])
                    
                    systemScoresAll['r1'].setdefault(systemId, {})[eventId] = rouge1Score
                    systemScoresAll['r2'].setdefault(systemId, {})[eventId] = rouge2Score
                    systemScoresAll['rL'].setdefault(systemId, {})[eventId] = rougeLScore
                except:
                    # in case the scores cannot be read well (means they are probably missing)
                    pass
    
    
    return systemScoresAll
    
def getOriginalScores(systemScoresAllOrig, eventIdsPerSystem):
    # put together lists of relevant original scores (only relevant systems and events):
    systemScoresAllToUse = {'pyr':{}, 'resp':{}, 'r1':{}, 'r2':{}, 'rL':{}} # { <'pyr'/'resp'/'r1'/'r2'/'rL'> -> { systemId -> [ <scores over relevant events> ] } }
    for systemId in eventIdsPerSystem:
        for eventId in eventIdsPerSystem[systemId]:
            systemScoresAllToUse['pyr'].setdefault(systemId, []).append(systemScoresAllOrig['pyr'][systemId][eventId])
            systemScoresAllToUse['resp'].setdefault(systemId, []).append(systemScoresAllOrig['resp'][systemId][eventId])
            systemScoresAllToUse['r1'].setdefault(systemId, []).append(systemScoresAllOrig['r1'][systemId][eventId])
            systemScoresAllToUse['r2'].setdefault(systemId, []).append(systemScoresAllOrig['r2'][systemId][eventId])
            systemScoresAllToUse['rL'].setdefault(systemId, []).append(systemScoresAllOrig['rL'][systemId][eventId])
            
    # get the system scores with the average of summary scores ({ <'pyr'/'resp'/'r1'/'r2'/'rL'> -> { systemId -> score } }):
    systemScoresToUse = {}
    systemScoresToUse['pyr'] = {systemId : reduce(lambda x, y: x + y, systemScoresAllToUse['pyr'][systemId]) / len(systemScoresAllToUse['pyr'][systemId]) \
        for systemId in systemScoresAllToUse['pyr']}
    systemScoresToUse['resp'] = {systemId : reduce(lambda x, y: x + y, systemScoresAllToUse['resp'][systemId]) / len(systemScoresAllToUse['resp'][systemId]) \
        for systemId in systemScoresAllToUse['resp']}
    systemScoresToUse['r1'] = {systemId : reduce(lambda x, y: x + y, systemScoresAllToUse['r1'][systemId]) / len(systemScoresAllToUse['r1'][systemId]) \
        for systemId in systemScoresAllToUse['r1']}
    systemScoresToUse['r2'] = {systemId : reduce(lambda x, y: x + y, systemScoresAllToUse['r2'][systemId]) / len(systemScoresAllToUse['r2'][systemId]) \
        for systemId in systemScoresAllToUse['r2']}
    systemScoresToUse['rL'] = {systemId : reduce(lambda x, y: x + y, systemScoresAllToUse['rL'][systemId]) / len(systemScoresAllToUse['rL'][systemId]) \
        for systemId in systemScoresAllToUse['rL']}
        
    return systemScoresToUse
    
    
def printAverageEventScores(summaryScores):
    # FOR DEBUGGING
    
    # list the summary scores in each topic:
    scoresByEvent = {eventId : [summaryScores[eventId][summId] for summId in summaryScores[eventId]] for eventId in summaryScores}
    # average the scores in each topic:
    avgScoreByEvent = {eventId : reduce(lambda x, y: x + y, scoresByEvent[eventId]) / len(scoresByEvent[eventId]) \
        for eventId in scoresByEvent}
    # print the average score of each topic:
    for eventId in avgScoreByEvent:
        print(eventId, avgScoreByEvent[eventId], len(scoresByEvent[eventId]))
    
    
    
def getSystemScores(summaryScores):
    # Gets the scores of the systems according to the separate summary scores of each system.
    # Returns a dictionary of scores per system and lists of eventIDs for which each system has a summary.
    
    # keep all the scores of each system:
    systemScoresAll = {} # the list of scores for each system.  { systemId -> [ <scores> ] }
    eventIdsUsedPerSystem = {} # the eventIds that each system has summaries for.  { systemId -> [ <event_ids> ] }
    for eventId in summaryScores:
        for summId in summaryScores[eventId]:
            systemId = summId.split('.')[-1] # the last part of the summary name is the system ID (e.g. D0601.M.250.A.4)
            systemScoresAll.setdefault(systemId, []).append(summaryScores[eventId][summId])
            
            # keep track of the eventIds used by the system:
            if not systemId in eventIdsUsedPerSystem:
                eventIdsUsedPerSystem[systemId] = [eventId]
            elif not eventId in eventIdsUsedPerSystem[systemId]:
                eventIdsUsedPerSystem[systemId].append(eventId)
            
    # get the system scores with the average of summary scores ({ systemId -> score }):
    systemScores = {systemId : reduce(lambda x, y: x + y, systemScoresAll[systemId]) / len(systemScoresAll[systemId]) \
        for systemId in systemScoresAll}
        
    return systemScores, eventIdsUsedPerSystem
    
    
def getSystemSummaryScores(dataValues, workersToFilter, questionIdsPerEvent, answerAggregationType, answerTieBreaker, numQuestionsPerSummary, numTurkersPerSummary, numEventsToUse, eventAgreements, percentEventsToFilter):
    # Get the score of each summary according to our lite-Pyramid method.
    # Returns a dictionary of { eventId -> { summId -> score } }.
    
    # first get the list of eventIds to use, according to the number specified:
    possibleEventIdsToUse = [eventId for eventId in dataValues.keys()]# if eventId not in eventsToFilter]
    if len(possibleEventIdsToUse) <= numEventsToUse:
        eventIdsToUse = possibleEventIdsToUse
    else:
        eventIdsToUse = random.sample(possibleEventIdsToUse, numEventsToUse)
    
    # if we need to filter out a certain percent of bad events, take them out of the eventIdsToUse:
    if percentEventsToFilter > 0:
        eventsToFilter = _filterEventsByAgreement(eventAgreements, percentEventsToFilter, baseEventIds=eventIdsToUse)
        eventIdsToUse = [eventId for eventId in eventIdsToUse if not eventId in eventsToFilter]
    
    #if len(dataValues.keys()) <= numEventsToUse:
    #    eventIdsToUse = dataValues.keys()
    #else:
    #    eventIdsToUse = random.sample(dataValues.keys(), numEventsToUse)
    
    # get a list of answers for each question in each summary:
    allSolutionsPerSummary = {} # { eventId -> { summaryId -> { questionId -> [ <full list of answers> ] } } }
    questionIdsToUse = {} # { eventId -> [ <questionIds> ] }
    for eventId in eventIdsToUse:
        questionIdsToUse[eventId] = []
        for summId in dataValues[eventId]:
            for solution in dataValues[eventId][summId]:
                # ignore this solution if this is a filtered worker:
                workerId = solution['workerId']
                if workerId in workersToFilter:
                    continue
                # add the questionId/answer to this summary:
                for questionId, answer in solution['answers'].items():
                    allSolutionsPerSummary.setdefault(eventId, {}).setdefault(summId, {}).setdefault(questionId, []).append(answer)
                    # keep the questionId for the event:
                    if questionId not in questionIdsToUse[eventId]:
                        questionIdsToUse[eventId].append(questionId)
                        
    # for each event, choose the sample of question IDs to use:
    for eventId in questionIdsToUse:
        if len(questionIdsToUse[eventId]) > numQuestionsPerSummary:
            questionIdsToUse[eventId] = random.sample(questionIdsToUse[eventId], numQuestionsPerSummary)
                
    # for each question to use, from the list of answers, choose a number of answers:
    summaryScores = {} # { eventId -> { summId -> score } }
    for eventId in allSolutionsPerSummary:
        summaryScores[eventId] = {}
        for summId in allSolutionsPerSummary[eventId]:
            summScore = 0.0 # the score is the sum of the questions' scores
            numQuestions = 0
            for questionId in allSolutionsPerSummary[eventId][summId]:
                # only look at the questions that should be used:
                if questionId in questionIdsToUse[eventId]:
                    # get a sample of answers for the question:
                    if len(allSolutionsPerSummary[eventId][summId][questionId]) <= numTurkersPerSummary:
                        answerSample = allSolutionsPerSummary[eventId][summId][questionId]
                    else:
                        answerSample = random.sample(allSolutionsPerSummary[eventId][summId][questionId], numTurkersPerSummary)
                    
                    # get the current question's score according to the several answers provided by the turkers:
                    questionFinalAnswerScore = getFinalAnswerScoreFromList(answerSample, answerAggregationType, answerTieBreaker)
                    summScore += questionFinalAnswerScore
                    numQuestions += 1
            
            # set the final score for the current summary as the percentage of positive answers:
            summaryScores[eventId][summId] = summScore / numQuestions
        
    return summaryScores
    

def getFinalAnswerScoreFromList(answerList, answerAggregationType, answerTieBreaker):
    # gets a score from the list of answers given (list of 0.0 or 1.0)
    # the score is a number between 0 and 1
    
    # use the average of the answers:
    if answerAggregationType == 0:
        result = reduce(lambda x, y: x + y, answerList) / len(answerList)
    # use the majority of the answers:
    elif answerAggregationType == 1:
        answersCount = Counter(answerList)
        if answersCount[0.0] > answersCount[1.0]:
            result = 0.0
        elif answersCount[0.0] < answersCount[1.0]:
            result = 1.0
        else:
            result = answerTieBreaker
    # return 1 iff atleast one 1, otherwise 0:
    elif answerAggregationType == 2:
        answersCount = Counter(answerList)
        if answersCount[1.0] > 0:
            result = 1.0
        else:
            result = 0.0
    else:
        result = -999
        
    return result

    

def getRawData(inputBatchFile):
    # get all the results from the MTurk batch results file:
    questionIdsPerEvent = {} # { eventId -> [questionIds] }
    rawDataValues = {} # { eventId -> { summId -> [{'workerId':<val> , 'answers':{questionId:<'p'/'n'/''>}}] } }
    with open(inputBatchFile, mode='r') as inF:
        csv_reader = csv.DictReader(inF)
        for row in csv_reader:
            eventId = row['Input.eventId']
            summId = row['Input.summaryId']
            workerId = row['WorkerId']
            questionIdList = ast.literal_eval(row['Input.qIdList'])
            
            # get the answers into a list, in order of the questionIdList (incremental index in the columns):
            answers = {qId:row['Answer.S{}Answer'.format(qInd+1)] for qInd, qId in enumerate(questionIdList)}
            
            # if there was no questions list added for this event yet:
            if not eventId in questionIdsPerEvent:
                questionIdsPerEvent[eventId] = questionIdList
                
            # if there's already a questions list for this event, and it doesn't contain the current list:
            elif not set(questionIdList) <= set(questionIdsPerEvent[eventId]):
                questionIdsPerEvent[eventId].extend(questionIdList) # extend the new list of questions

            rawDataValues.setdefault(eventId, {}).setdefault(summId, []).append({'workerId':workerId, 'answers':answers})
            
    return rawDataValues, questionIdsPerEvent


def mapValues(rawDataValues, noAnswerDefaultValue):
    # Maps the raw data to values for use, also according to the configuration given.
    # Returns { eventId -> { summId -> [{'workerId':<val> , 'answers':{qId:<0/1>}}] } }.

    # the inner function to map a raw value to a processable value:
    def mapFunc(sourceVal):
        if sourceVal == 'p':
            return 1.0
        elif sourceVal == 'n':
            return 0.0
        elif sourceVal == '':
            return noAnswerDefaultValue

    # fully copy the raw data:
    mappedValues = copy.deepcopy(rawDataValues) # { eventId -> { summId -> [{'workerId':<val> , 'answers':{qId:<0/1>}}] } }
    for eventId in mappedValues:
        for summId in mappedValues[eventId]:
            for solution in mappedValues[eventId][summId]:
                for qId in solution['answers']:
                    # replace the raw value to the new value:
                    solution['answers'][qId] = mapFunc(solution['answers'][qId])

    return mappedValues
    
def getWorkersToFilter(dataValues, questionIdsPerEvent, workerAgreementThreshold, numFilteringIteration, printToScreen=False):
    # Gets a list of workerIDs to ignore due to low agreement with others.
    
    workersToFilter = []
    
    for iter in range(numFilteringIteration):
        # measure the worker agreements:
        workerAgreements, workerAssignmentsCount = _measureWorkerAgreement(dataValues, questionIdsPerEvent, workersToFilter)
        if printToScreen:
            for workerId in workerAgreements:
                print '{}\t{}\t{}'.format(workerId, workerAgreements[workerId], workerAssignmentsCount[workerId])
            
        # get the workers to filter due to low agreement scores:
        workersToFilter = _filterWorkersByAgreement(workerAgreements, workerAssignmentsCount, workerAgreementThreshold)
        if printToScreen:
            for workerId in workersToFilter:
                print 'Filtered: {}\t{}\t{}'.format(workerId, workerAgreements[workerId], workerAssignmentsCount[workerId])
        
    return workersToFilter
                    
def _measureWorkerAgreement(dataValues, questionIdsPerEvent, workerIgnoreList):
    # Gets the agreement scores of each worker.
    # Returns workerAgreements {workerId -> overall agreement score} and workerAssignmentsCount {workerId -> # of assignments done}.
        
    def calculateAgreement(list1, list2):
        # get the percentage agreement between the two answer lists:
        seqenceObj = difflib.SequenceMatcher(None, answers_i, answers_j)
        agreementScore = seqenceObj.ratio()
        return agreementScore
    
    workerAgreementDict = {} # workerId -> [<agreement values>]
    workerAssignmentsCount = {} # workerId -> # of assignments done
    
    for eventId in dataValues:
        for summId in dataValues[eventId]:
            
            # in the summary, there are several solutions, so go over each pair of solutions (questionnaires) and measure agreement:
            for i in range(len(dataValues[eventId][summId])):
                
                solution_i = dataValues[eventId][summId][i] # i_th solution of the current summary
                workerId_i = solution_i['workerId']
                
                # check whether to ignore this worker:
                if workerId_i in workerIgnoreList:
                    continue
                
                questionIds = solution_i['answers'].keys()
                answers_i = [solution_i['answers'][qId] for qId in questionIds]
                
                for j in range(i+1, len(dataValues[eventId][summId])):
                    solution_j = dataValues[eventId][summId][j] # j_th solution of the current summary
                    workerId_j = solution_j['workerId']
                    
                    # check whether to ignore this worker:
                    if workerId_j in workerIgnoreList:
                        continue
                        
                    # if the two solutions (assignments) have the same question ID sets, 
                    #   measure the agreement between the two annotators:
                    if set(solution_i['answers'].keys()) == set(solution_j['answers'].keys()):
                        answers_j = [solution_j['answers'][qId] for qId in questionIds]
                        
                        agreementScore = calculateAgreement(answers_i, answers_j)
                
                        # add the agreement score to each of the workers:
                        workerAgreementDict.setdefault(workerId_i, []).append(agreementScore)
                        workerAgreementDict.setdefault(workerId_j, []).append(agreementScore)
                        
                # keep count of the number of assignments done by the worker:
                workerAssignmentsCount[workerId_i] = workerAssignmentsCount.get(workerId_i, 0) + 1

    # calculate the average agreement accuracy for each worker:
    workerAgreements = {} # workerId -> overall agreement score
    for workerId in workerAgreementDict:
        workerAgreements[workerId] = round(reduce(lambda x, y: x + y, workerAgreementDict[workerId]) / len(workerAgreementDict[workerId]), 3)
        
    return workerAgreements, workerAssignmentsCount

    
def _filterWorkersByAgreement(workerAgreements, workerAssignmentsCount, workerAgreementThreshold):
    return [workerId for workerId in workerAgreements if workerAgreements[workerId] < workerAgreementThreshold]
    

def getEventsToFilter(dataValues, workersToFilter, percentEventsToFilter, printToScreen=False):
    # Gets a list of events to filter (percentEventsToFilter), ordered from highest disagreeing to least.
    
    # measure the event agreements:
    eventAgreements = _measureEventAgreement(dataValues, workersToFilter)
    if printToScreen:
        for eventId in eventAgreements:
            print '{}\t{}'.format(eventId, eventAgreements[eventId])
        
    # get the events to filter due to low agreement scores:
    eventsToFilter = _filterEventsByAgreement(eventAgreements, percentEventsToFilter)
    if printToScreen:
        for eventId in eventsToFilter:
            print 'Filtered: {}\t{}'.format(eventId, eventAgreements[eventId])
    
    return eventsToFilter
    
def _measureEventAgreement(dataValues, workersToFilter):
    MISSING_VALUE_CHAR = '*' # the character signaling an ungiven answer
    
    def calculateAgreement(list2d):
        return krippendorff_alpha(list2d, convert_items=str, missing_items=[MISSING_VALUE_CHAR])
    
    eventAgreementDict = {} # eventId -> [<agreement values over systems summaries>]
    
    for eventId in dataValues:
        currentEventAgreementScores = []
        for summId in dataValues[eventId]:
            allAnswers = {} # { questionSet -> [<list of q/a dictionaries>] }
            # in the summary, there are several solutions, so go over each pair of solutions (questionnaires) and measure agreement:
            for i in range(len(dataValues[eventId][summId])):
                solution_i = dataValues[eventId][summId][i] # i_th solution of the current summary
                worker_i = solution_i['workerId']
                if worker_i in workersToFilter:
                    continue
                questionIdsStr = str(solution_i['answers'].keys()) # a string to represent the question set (2 per summary)
                qaDictCopy = {qId : answer if answer != '' else MISSING_VALUE_CHAR for qId, answer in solution_i['answers'].items()} # replace '' with '*'
                allAnswers.setdefault(questionIdsStr, []).append(qaDictCopy)
                
            # for each question set of this summary, get the agreement score:
            for qSet in allAnswers:
                agreementScore = calculateAgreement(allAnswers[qSet])
                currentEventAgreementScores.append(agreementScore)
        
        # keep the average agreement score 
        eventAgreementDict[eventId] = reduce(lambda x,y:x+y, currentEventAgreementScores) / len(currentEventAgreementScores)
                

    ## calculate the average agreement accuracy for each event:
    #eventAgreements = {} # eventId -> overall agreement score
    #for eventId in eventAgreementDict:
    #    eventAgreements[eventId] = round(reduce(lambda x, y: x + y, eventAgreementDict[eventId]) / len(eventAgreementDict[eventId]), 3)
        
    return eventAgreementDict

def _filterEventsByAgreement(eventAgreements, percentEventsToFilter, baseEventIds=None):
    # if needed, prepare a list of eventIds to use according to the base list given:
    if baseEventIds == None:
        baseEventAgreements = eventAgreements
    else:
        baseEventAgreements = {evId:agr for evId, agr in eventAgreements.items() if evId in baseEventIds}
        
    # sort the events by agreement:
    sortedByAgreement = sorted(baseEventAgreements.items(), key=operator.itemgetter(1))
    # get the number of events to leave out:
    numEventsToFilter = int(float(len(baseEventAgreements)) * percentEventsToFilter)
    # get the evemts to leave out (lowest agreement scores):
    eventIdsToFilter = [eventId for eventId, _ in sortedByAgreement[0:numEventsToFilter]]
    
    return eventIdsToFilter
    
    
# Print iterations progress
def _printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '$'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    sys.stdout.write('\r{} |{}| {}% {}'.format(prefix, bar, percent, suffix))
    # Print New Line on Complete
    if iteration == total: 
        print()

### krippendorff_alpha START ###

def nominal_metric(a, b):
    return a != b


def interval_metric(a, b):
    return (a-b)**2


def ratio_metric(a, b):
    return ((a-b)/(a+b))**2


def krippendorff_alpha(data, metric=nominal_metric, force_vecmath=False, convert_items=float, missing_items=None):
    '''
    From: https://github.com/grrrr/krippendorff-alpha
    Calculate Krippendorff's alpha (inter-rater reliability):
    
    data is in the format
    [
        {unit1:value, unit2:value, ...},  # coder 1
        {unit1:value, unit3:value, ...},   # coder 2
        ...                            # more coders
    ]
    or 
    it is a sequence of (masked) sequences (list, numpy.array, numpy.ma.array, e.g.) with rows corresponding to coders and columns to items
    
    metric: function calculating the pairwise distance
    force_vecmath: force vector math for custom metrics (numpy required)
    convert_items: function for the type conversion of items (default: float)
    missing_items: indicator for missing items (default: None)
    '''
    
    # number of coders
    m = len(data)
    
    # set of constants identifying missing values
    if missing_items is None:
        maskitems = []
    else:
        maskitems = list(missing_items)
    if np is not None:
        maskitems.append(np.ma.masked_singleton)
    
    # convert input data to a dict of items
    units = {}
    for d in data:
        try:
            # try if d behaves as a dict
            diter = d.items()
        except AttributeError:
            # sequence assumed for d
            diter = enumerate(d)
            
        for it, g in diter:
            if g not in maskitems:
                try:
                    its = units[it]
                except KeyError:
                    its = []
                    units[it] = its
                its.append(convert_items(g))


    units = dict((it, d) for it, d in units.items() if len(d) > 1)  # units with pairable values
    n = sum(len(pv) for pv in units.values())  # number of pairable values
    
    if n == 0:
        raise ValueError("No items to compare.")
    
    np_metric = (np is not None) and ((metric in (interval_metric, nominal_metric, ratio_metric)) or force_vecmath)
    
    Do = 0.
    for grades in units.values():
        if np_metric:
            gr = np.asarray(grades)
            Du = sum(np.sum(metric(gr, gri)) for gri in gr)
        else:
            Du = sum(metric(gi, gj) for gi in grades for gj in grades)
        Do += Du/float(len(grades)-1)
    Do /= float(n)

    if Do == 0:
        return 1.

    De = 0.
    for g1 in units.values():
        if np_metric:
            d1 = np.asarray(g1)
            for g2 in units.values():
                De += sum(np.sum(metric(d1, gj)) for gj in g2)
        else:
            for g2 in units.values():
                De += sum(metric(gi, gj) for gi in g1 for gj in g2)
    De /= float(n*(n-1))

    return 1.-Do/De if (Do and De) else 1.
    
### krippendorff_alpha END ###
    
if __name__ == '__main__':
    
    ONLY_SCORES = True
    if len(sys.argv) > 1:
        if sys.argv[1] == '-scores':
            ONLY_SCORES = True
        elif sys.argv[1] == '-corr':
            ONLY_SCORES = False
        else:
            print('Usage: calculateScores.py [-scores|-corr]')
            
    
    # get the raw data from the MTurk batch output:
    rawDataValues, questionIdsPerEvent = getRawData(RESULTS_FILE_INPUT)
    
    # get the original manual scores per systemId and eventId from the manual scores file:
    systemScoresAllOrig = readOriginalScoresData(MANUAL_SCORES_FILE, ROUGE_SCORES_FILE)
    
    # get the number of configurations being tested (just for the progress bar in the CLI):
    numConfigurations = len(ANSWER_AGGREGATION_TYPE) * len(ANSWER_TIE_BREAKER) * len(NO_ANSWER_DEFAULT) * len(NUM_TURKERS_PER_SUMMARY) * len(NUM_QUESTIONS_PER_SUMMARY) * len(WORKER_AGREEMENT_THRESHOLD) * len(AGREEMENT_FILTERING_ITERATIONS) * len(NUM_EVENTS_TO_USE) * len(EVENT_FILTER_PERCENT)
    
    # for measuring time:
    startTime = time.time()
    
    # calculate the correlations between the real pyramids and our methodology, and write to output file:
    with open(OUTPUT_FILE, 'w') as outF:
        
        # write out the column names in the first row:
        if not ONLY_SCORES:
            outF.write('{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n'.format(
                'ANSWER_AGGREGATION_TYPE',
                'ANSWER_TIE_BREAKER',
                'NO_ANSWER_DEFAULT',
                'NUM_TURKERS_PER_SUMMARY',
                'NUM_QUESTIONS_PER_SUMMARY',
                'NUM_EVENTS_TO_USE',
                'WORKER_AGREEMENT_THRESHOLD',
                'AGREEMENT_FILTERING_ITERATIONS',
                'EVENT_FILTER_PERCENT',
                'pearsonCorr', 'pearsonCorrStd', 'pearsonPVal', 'spearmanCorr', 'spearmanCorrStd', 'spearmanPVal', 'rankOrig', 'rankOurs',
                'pCorrResp', 'pPvalResp', 'sCorrResp', 'sPvalResp',
                'pCorrR1', 'pPvalR1', 'sCorrR1', 'sPvalR1',
                'pCorrR2', 'pPvalR2', 'sCorrR2', 'sPvalR2',
                'pCorrRL', 'pPvalRL', 'sCorrRL', 'sPvalRL'))
        else:
            outF.write('{},{},{},{},{},{},{},{},{},{},{}\n'.format(
                'ANSWER_AGGREGATION_TYPE',
                'ANSWER_TIE_BREAKER',
                'NO_ANSWER_DEFAULT',
                'NUM_TURKERS_PER_SUMMARY',
                'NUM_QUESTIONS_PER_SUMMARY',
                'NUM_EVENTS_TO_USE',
                'WORKER_AGREEMENT_THRESHOLD',
                'AGREEMENT_FILTERING_ITERATIONS',
                'EVENT_FILTER_PERCENT',
                'rankOrig', 'rankOurs'))
        
        # do a "grid search" over all configurations:
        _printProgressBar(0, numConfigurations, prefix = 'Progress:', suffix = '', length = 50)
        confugurationNum = 1
        for answerAggregationType in ANSWER_AGGREGATION_TYPE:
            for answerTieBreaker in ANSWER_TIE_BREAKER:
                for noAnswerDefaultValue in NO_ANSWER_DEFAULT:
                    for numTurkersPerSummary in NUM_TURKERS_PER_SUMMARY:
                        for numQuestionsPerSummary in NUM_QUESTIONS_PER_SUMMARY:
                            for numEventsToUse in NUM_EVENTS_TO_USE:
                                for workerAgreementThreshold in WORKER_AGREEMENT_THRESHOLD:
                                    for agreementFilteringIterations in AGREEMENT_FILTERING_ITERATIONS:
                                        for eventFilterPercent in EVENT_FILTER_PERCENT:
                                            
                                            # set the current configuration paramaters:
                                            configuration = {
                                                'ANSWER_AGGREGATION_TYPE' : answerAggregationType,
                                                'ANSWER_TIE_BREAKER' : answerTieBreaker,
                                                'NO_ANSWER_DEFAULT' : noAnswerDefaultValue,
                                                'NUM_TURKERS_PER_SUMMARY' : numTurkersPerSummary,
                                                'NUM_QUESTIONS_PER_SUMMARY' : numQuestionsPerSummary,
                                                'NUM_EVENTS_TO_USE' : numEventsToUse,
                                                'WORKER_AGREEMENT_THRESHOLD' : workerAgreementThreshold,
                                                'AGREEMENT_FILTERING_ITERATIONS' : agreementFilteringIterations,
                                                'NUM_ITERATION_ON_CONFIGURATION' : NUM_ITERATION_ON_CONFIGURATION,
                                                'EVENT_FILTER_PERCENT' : eventFilterPercent
                                            }
                                            
                                            # get the correlations for the current configuration:
                                            pearsonCorr, pearsonCorrStd, pearsonPVal, spearmanCorr, spearmanCorrStd, spearmanPVal, \
                                                systemScoresOurs, systemScoresOriginal, \
                                                pearsonCorrOrig, pearsonPValueOrig, spearmanCorrOrig, spearmanPValueOrig = \
                                                computeScoresAndCorrelations(rawDataValues, questionIdsPerEvent, systemScoresAllOrig, configuration, ONLY_SCORES)
                                            
                                            # show the progress and time after the running on the configuration:
                                            curTime = time.time() - startTime
                                            _printProgressBar(confugurationNum, numConfigurations, prefix = 'Progress:', suffix = round(curTime/60,2), length = 50)
                                            confugurationNum += 1
                                            
                                            # write out the configuration paramaters and the correlations:
                                            if not ONLY_SCORES:
                                                lineToOutput = '{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n'.format(
                                                    answerAggregationType,
                                                    answerTieBreaker,
                                                    noAnswerDefaultValue,
                                                    numTurkersPerSummary,
                                                    numQuestionsPerSummary,
                                                    numEventsToUse,
                                                    workerAgreementThreshold,
                                                    agreementFilteringIterations,
                                                    eventFilterPercent,
                                                    pearsonCorr, pearsonCorrStd, pearsonPVal, spearmanCorr, spearmanCorrStd, spearmanPVal,
                                                    ' '.join('{}:{}'.format(sysId, sysScore) for sysId, sysScore in systemScoresOriginal.items()),
                                                    ' '.join('{}:{}'.format(sysId, sysScore) for sysId, sysScore in systemScoresOurs.items()),
                                                    pearsonCorrOrig['resp'], pearsonPValueOrig['resp'], spearmanCorrOrig['resp'], spearmanPValueOrig['resp'],
                                                    pearsonCorrOrig['r1'], pearsonPValueOrig['r1'], spearmanCorrOrig['r1'], spearmanPValueOrig['r1'],
                                                    pearsonCorrOrig['r2'], pearsonPValueOrig['r2'], spearmanCorrOrig['r2'], spearmanPValueOrig['r2'],
                                                    pearsonCorrOrig['rL'], pearsonPValueOrig['rL'], spearmanCorrOrig['rL'], spearmanPValueOrig['rL'])
                                            else:
                                                lineToOutput = '{},{},{},{},{},{},{},{},{},{},{}\n'.format(
                                                    answerAggregationType,
                                                    answerTieBreaker,
                                                    noAnswerDefaultValue,
                                                    numTurkersPerSummary,
                                                    numQuestionsPerSummary,
                                                    numEventsToUse,
                                                    workerAgreementThreshold,
                                                    agreementFilteringIterations,
                                                    eventFilterPercent,
                                                    ' '.join('{}:{}'.format(sysId, sysScore) for sysId, sysScore in systemScoresOriginal.items()),
                                                    ' '.join('{}:{}'.format(sysId, sysScore) for sysId, sysScore in systemScoresOurs.items()))
                                                    
                                            outF.write(lineToOutput)