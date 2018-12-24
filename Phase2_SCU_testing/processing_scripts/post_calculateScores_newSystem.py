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
This script gets the scores of a system according to the Lite-Pyramid evaluation method, based on crowdsourced SCU judgments.
Run: python Phase2_SCU_testing/processing_scripts/post_calculateScores_newSystem.py <path_to_AMT_results_file>

If the crowdsourced task was run more than once, combine the two results files from AMT into one file (don't copy the header line from one file to the other).
Make sure there's only one system evaluated in the results file, since all results are taken into account.
'''

try:
    RESULTS_FILE_INPUT = sys.argv[1]
except:
    print('Usage: python Phase2_SCU_testing/processing_scripts/post_calculateScores_newSystem.py <path_to_AMT_results_file> <2005|2006>')


### Configuration options:
# How to decide on an answer when several are provided
ANSWER_AGGREGATION_TYPE = 1 # 0=average 1=majority 2=atleast1present
# If there happens to be an even number of answers and there's a tie, what is the answer:
ANSWER_TIE_BREAKER = 0.0 # 0.0=NotPresent, 1.0=Present, 0.5=Gives half a point
# If there's no given answer by a turker, what value should it be assigned:
NO_ANSWER_DEFAULT = 1.0 # 0.0=NotPresent, 1.0=Present, 0.5=Ignore
# The number of turkers (assignments) to use per summary:
NUM_TURKERS_PER_SUMMARY = 5 # if more than this number, takes random sample, if less, takes all
# The number of questions to use for each summary evaluation
NUM_QUESTIONS_PER_SUMMARY = 32 # if more than this number, takes random sample (consistent over event), if less, takes all
# The number of events on which to evaluate:
NUM_EVENTS_TO_USE = 20 # if more than this number, takes random sample, if less, takes all
# The agreement threshold below which to disregard a workers answers:
WORKER_AGREEMENT_THRESHOLD = 0.5 # if the agreement of a worker is below this threshold, then the worker is filtered out
## TODO: take into account how many assignments the worker has done
# How many times should agreement filtering be done (after removing some, filter again on the remaining):
AGREEMENT_FILTERING_ITERATIONS = 1
# What percent of the events to filter by agreement of event answers:
EVENT_FILTER_PERCENT = 0.0
# How many times should each configuration be tested and then averaged:
NUM_ITERATION_ON_CONFIGURATION = 70


def computeScores(rawDataValues, questionIdsPerEvent, configuration, startTime):
    
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
    _printProgressBar(0, configuration['NUM_ITERATION_ON_CONFIGURATION'], prefix = 'Progress:', suffix = '', length = 50)
    
    summaryScoresPerEventAll = {}
    systemScoreAll = []
    for i in range(configuration['NUM_ITERATION_ON_CONFIGURATION']):
        
        # get a list of events to disregard during scoring:
        #eventsToFilter = getEventsToFilter(
        #    rawDataValues,
        #    workersToFilter,
        #    configuration['EVENT_FILTER_PERCENT'],
        #    printToScreen=False)
        #eventsToFilter = _filterEventsByAgreement(eventAgreements, )
        
        # get the scores of each system summary (per event):
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
        
        # keep this iteration's score per eventId:
        for eventId in summaryScores:
            summaryScoresPerEventAll.setdefault(eventId, []).append(summaryScores[eventId])
        
        # get the system scores (in our method) according to their summary scores:
        systemScore, eventIdsUsed = getSystemScore(summaryScores)
        
        # keep this iteration's system score (average over events):
        systemScoreAll.append(systemScore)
        
        # show the progress and time after the running on the configuration:
        curTime = time.time() - startTime
        _printProgressBar(i, configuration['NUM_ITERATION_ON_CONFIGURATION'], prefix = 'Progress:', suffix = round(curTime/60,2), length = 50)
    
    _printProgressBar(configuration['NUM_ITERATION_ON_CONFIGURATION'], configuration['NUM_ITERATION_ON_CONFIGURATION'], prefix = 'Progress:', suffix = round(curTime/60,2), length = 50)
    
    # the average scores per eventId over all iterations:
    summaryScorePerEvent = {} # { eventId -> score }
    for eventId in summaryScoresPerEventAll:
        summaryScorePerEvent[eventId] = reduce(lambda x, y: x + y, summaryScoresPerEventAll[eventId]) / len(summaryScoresPerEventAll[eventId])
    
    # now that we've finished running many iterations, calculate the average system scores over the iterations:
    systemScoreFinal = reduce(lambda x, y: x + y, systemScoreAll) / len(systemScoreAll)
    
    return summaryScorePerEvent, systemScoreFinal
    

    
    
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
    
    
    
def getSystemScore(summaryScores):
    # Gets the average score of the system.
    # Returns the final score and the list of eventIDs for which the system has a summary.
    
    systemScoresAll = summaryScores.values() # the list of scores of the system.  [ <scores> ]
    eventIdsUsed = summaryScores.keys() # the eventIds that the system has summaries for. [ <event_ids> ]
            
    # get the system scores with the average of summary scores:
    systemScore = reduce(lambda x, y: x + y, systemScoresAll) / len(systemScoresAll)
        
    return systemScore, eventIdsUsed
    
    
def getSystemSummaryScores(dataValues, workersToFilter, questionIdsPerEvent, answerAggregationType, answerTieBreaker, numQuestionsPerSummary, numTurkersPerSummary, numEventsToUse, eventAgreements, percentEventsToFilter):
    # Get the score of each summary according to our lite-Pyramid method.
    # Returns a dictionary of { eventId -> score }.
    
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
        
        for solution in dataValues[eventId]:
            # ignore this solution if this is a filtered worker:
            workerId = solution['workerId']
            if workerId in workersToFilter:
                continue
            # add the questionId/answer to this summary:
            for questionId, answer in solution['answers'].items():
                allSolutionsPerSummary.setdefault(eventId, {}).setdefault(questionId, []).append(answer)
                # keep the questionId for the event:
                if questionId not in questionIdsToUse[eventId]:
                    questionIdsToUse[eventId].append(questionId)
                        
    # for each event, choose the sample of question IDs to use:
    for eventId in questionIdsToUse:
        if len(questionIdsToUse[eventId]) > numQuestionsPerSummary:
            questionIdsToUse[eventId] = random.sample(questionIdsToUse[eventId], numQuestionsPerSummary)
                
    # for each question to use, from the list of answers, choose a number of answers:
    summaryScores = {} # { eventId -> score } }
    for eventId in allSolutionsPerSummary:
        summScore = 0.0 # the score is the sum of the questions' scores
        numQuestions = 0
        for questionId in allSolutionsPerSummary[eventId]:
            # only look at the questions that should be used:
            if questionId in questionIdsToUse[eventId]:
                # get a sample of answers for the question:
                if len(allSolutionsPerSummary[eventId][questionId]) <= numTurkersPerSummary:
                    answerSample = allSolutionsPerSummary[eventId][questionId]
                else:
                    answerSample = random.sample(allSolutionsPerSummary[eventId][questionId], numTurkersPerSummary)
                
                # get the current question's score according to the several answers provided by the turkers:
                questionFinalAnswerScore = getFinalAnswerScoreFromList(answerSample, answerAggregationType, answerTieBreaker)
                summScore += questionFinalAnswerScore
                numQuestions += 1
        
        # set the final score for the current summary as the percentage of positive answers:
        summaryScores[eventId] = summScore / numQuestions
        
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
    rawDataValues = {} # { eventId -> [{'workerId':<val> , 'answers':{questionId:<'p'/'n'/''>}}] }
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

            rawDataValues.setdefault(eventId, []).append({'workerId':workerId, 'answers':answers})
            
    return rawDataValues, questionIdsPerEvent


def mapValues(rawDataValues, noAnswerDefaultValue):
    # Maps the raw data to values for use, also according to the configuration given.
    # Returns { eventId -> [{'workerId':<val> , 'answers':{qId:<0/1>}}] } }.

    # the inner function to map a raw value to a processable value:
    def mapFunc(sourceVal):
        if sourceVal == 'p':
            return 1.0
        elif sourceVal == 'n':
            return 0.0
        elif sourceVal == '':
            return noAnswerDefaultValue

    # fully copy the raw data:
    mappedValues = copy.deepcopy(rawDataValues) # { eventId -> [{'workerId':<val> , 'answers':{qId:<0/1>}}] }
    for eventId in mappedValues:
        for solution in mappedValues[eventId]:
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
        # in the summary, there are several solutions, so go over each pair of solutions (questionnaires) and measure agreement:
        for i in range(len(dataValues[eventId])):
            
            solution_i = dataValues[eventId][i] # i_th solution of the current summary
            workerId_i = solution_i['workerId']
            
            # check whether to ignore this worker:
            if workerId_i in workerIgnoreList:
                continue
            
            questionIds = solution_i['answers'].keys()
            answers_i = [solution_i['answers'][qId] for qId in questionIds]
            
            for j in range(i+1, len(dataValues[eventId])):
                solution_j = dataValues[eventId][j] # j_th solution of the current summary
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
        
        allAnswers = {} # { questionSet -> [<list of q/a dictionaries>] }
        # in the summary, there are several solutions, so go over each pair of solutions (questionnaires) and measure agreement:
        for i in range(len(dataValues[eventId])):
            solution_i = dataValues[eventId][i] # i_th solution of the current summary
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
    
    # for measuring time:
    startTime = time.time()
    
    # get the raw data from the MTurk batch output:
    rawDataValues, questionIdsPerEvent = getRawData(RESULTS_FILE_INPUT)
    
    # set the current configuration paramaters:
    configuration = {
        'ANSWER_AGGREGATION_TYPE' : ANSWER_AGGREGATION_TYPE,
        'ANSWER_TIE_BREAKER' : ANSWER_TIE_BREAKER,
        'NO_ANSWER_DEFAULT' : NO_ANSWER_DEFAULT,
        'NUM_TURKERS_PER_SUMMARY' : NUM_TURKERS_PER_SUMMARY,
        'NUM_QUESTIONS_PER_SUMMARY' : NUM_QUESTIONS_PER_SUMMARY,
        'NUM_EVENTS_TO_USE' : NUM_EVENTS_TO_USE,
        'WORKER_AGREEMENT_THRESHOLD' : WORKER_AGREEMENT_THRESHOLD,
        'AGREEMENT_FILTERING_ITERATIONS' : AGREEMENT_FILTERING_ITERATIONS,
        'NUM_ITERATION_ON_CONFIGURATION' : NUM_ITERATION_ON_CONFIGURATION,
        'EVENT_FILTER_PERCENT' : EVENT_FILTER_PERCENT
    }
                                            
    # get the scores for the current configuration:
    summaryScorePerEvent, systemScoreFinal = computeScores(rawDataValues, questionIdsPerEvent, configuration, startTime)
    
    # print out the scores:
    print
    for eventId in summaryScorePerEvent:
        print('{}\t{}'.format(eventId, summaryScorePerEvent[eventId]))
    print('Final score: {}'.format(systemScoreFinal))