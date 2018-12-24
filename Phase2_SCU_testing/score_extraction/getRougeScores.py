'''
This script reads the different ROUGE scores from the DUC 2005 and 2006 scores files.
It outputs the scores to a CSV file in a format to be used when calaculating correlations in the phase2 post-script.

Provide the input and output files.
'''

'''
The original scores file.
2005: e.g. '../../../DUC_data/2005/results/ROUGE/rougejk.m.out'
2006: e.g. '../../../DUC_data/2006/NISTeval2/ROUGE/rougejk.m.out'
'''
ROUGE_SCORES_FILE = ''
'''
The file to output to.
'''
OUTPUT_CSV_FILE = '' # e.g. '2006RougeScoresAvg.csv'




START = 0
NEWROUGE = 1
SCORES = 2
m_lineState = START

m_valuesDict = {}
m_rougeVariants = []
m_scoresByEvent = {}

def main(rougeScoresFile, outputCsvFile):
    global m_lineState
    
    # read in the scores:
    with open(rougeScoresFile, 'r') as inF:
        for line in inF:
            line = line.strip()
            
            if line.startswith('---'):
                m_lineState = NEWROUGE
            elif line.startswith('...'):
                m_lineState = SCORES
            else:
                if m_lineState == NEWROUGE:
                    processLineNewRouge(line)
                elif m_lineState == SCORES:
                    processLineScores(line)
    
    # average the scores:
    averageAll()
    # print out the final scores:
    printStats(outputCsvFile)

def printStats(outputCsvFile):
    metrics = ['recall', 'precision', 'f1']
    with open(outputCsvFile, 'w') as outF:
        # first print the average system scores over all events:
        outF.write('systemId')
        for rougeVariant in m_rougeVariants:
            for metric in metrics:
                outF.write(', {0} {1}'.format(rougeVariant, metric))
        outF.write('\n')
        
        for systemId in m_valuesDict:
            outF.write(systemId)
            for rougeVariant in m_rougeVariants:
                for metric in metrics:
                    outF.write(', ' + m_valuesDict[systemId][rougeVariant][metric]['avg'])
            outF.write('\n')
            
        # print the average scores in each event:
        outF.write('\n\n')
        outF.write('eventId')
        for rougeVariant in m_rougeVariants:
            for metric in metrics:
                outF.write(', {0} {1}'.format(rougeVariant, metric))
        outF.write('\n')
        for eventId in m_scoresByEvent:
            outF.write(eventId)
            for rougeVariant in m_rougeVariants:
                for metric in metrics:
                    outF.write(', ' + str(m_scoresByEvent[eventId][rougeVariant][metric]['avg']))
            outF.write('\n')
        
        # print the system scores for each event:
        outF.write('\n\n')
        outF.write('systemId, eventId ')
        for rougeVariant in m_rougeVariants:
            for metric in metrics:
                outF.write(', {0} {1}'.format(rougeVariant, metric))
        for systemId in m_valuesDict:
            for eventId in m_scoresByEvent:
                outF.write('{}, {}'.format(systemId, eventId))
                for rougeVariant in m_rougeVariants:
                    for metric in metrics:
                        if eventId in m_valuesDict[systemId][rougeVariant][metric]:
                            outF.write(', {}'.format(m_valuesDict[systemId][rougeVariant][metric][eventId]['avg']))
                        else:
                            outF.write(', ')
                outF.write('\n')
        
        
    
    
def averageAll():
    global m_valuesDict
    global m_scoresByEvent
    
    eventIds = {} # keep a list of the eventIds
    for systemId in m_valuesDict:
        for rougeVariant in m_valuesDict[systemId]:
            for metric in m_valuesDict[systemId][rougeVariant]:
                for eventId in m_valuesDict[systemId][rougeVariant][metric]:
                    eventIds[eventId] = 1
                    if eventId != 'avg':
                        scoresOverReferences = m_valuesDict[systemId][rougeVariant][metric][eventId]['vals']
                        m_valuesDict[systemId][rougeVariant][metric][eventId]['avg'] = reduce(lambda x, y: x + y, scoresOverReferences) / len(scoresOverReferences)
                        
    # get all the scores of each event:
    m_scoresByEvent = {eventId:{} for eventId in eventIds if eventId != 'avg'}
    for systemId in m_valuesDict:
        for rougeVariant in m_valuesDict[systemId]:
            for metric in m_valuesDict[systemId][rougeVariant]:
                for eventId in m_valuesDict[systemId][rougeVariant][metric]:
                    if eventId == 'avg':
                        continue
                    if not rougeVariant in m_scoresByEvent[eventId]:
                        m_scoresByEvent[eventId][rougeVariant] = {}
                    if not metric in m_scoresByEvent[eventId][rougeVariant]:
                        m_scoresByEvent[eventId][rougeVariant][metric] = {'vals':[], 'avg':-1}
                    if eventId != 'avg':
                        m_scoresByEvent[eventId][rougeVariant][metric]['vals'].extend(m_valuesDict[systemId][rougeVariant][metric][eventId]['vals'])
    
    # average the scores for each event:
    for eventId in m_scoresByEvent:
        for rougeVariant in m_scoresByEvent[eventId]:
            for metric in m_scoresByEvent[eventId][rougeVariant]:
                m_scoresByEvent[eventId][rougeVariant][metric]['avg'] = reduce(lambda x, y: x + y, m_scoresByEvent[eventId][rougeVariant][metric]['vals']) / len(m_scoresByEvent[eventId][rougeVariant][metric]['vals'])
                        
def processLineScores(line):
    global m_valuesDict
    
    systemId, rougeVariant, _, comparedSummIds, recStr, precStr, f1Str = line.split()
    eventId = comparedSummIds.split('.')[0]
    recVal, precVal, f1Val = recStr.split(':')[1], precStr.split(':')[1], f1Str.split(':')[1]
    
    if not eventId in m_valuesDict[systemId][rougeVariant]['recall']:
        m_valuesDict[systemId][rougeVariant]['recall'][eventId] = {'vals':[], 'avg':-1}
        m_valuesDict[systemId][rougeVariant]['precision'][eventId] = {'vals':[], 'avg':-1}
        m_valuesDict[systemId][rougeVariant]['f1'][eventId] = {'vals':[], 'avg':-1}
    
    m_valuesDict[systemId][rougeVariant]['recall'][eventId]['vals'].append(float(recVal))
    m_valuesDict[systemId][rougeVariant]['precision'][eventId]['vals'].append(float(precVal))
    m_valuesDict[systemId][rougeVariant]['f1'][eventId]['vals'].append(float(f1Val))

def processLineNewRouge(line):
    global m_valuesDict
    global m_rougeVariants

    lineParts = line.split()
    systemId, rougeVariant, metric, value = lineParts[0:4]
    
    if metric == 'Average_R:':
        metric = 'recall'
    elif metric == 'Average_P:':
        metric = 'precision'
    elif metric == 'Average_F:':
        metric = 'f1'
    
    if not systemId in m_valuesDict:
        m_valuesDict[systemId] = {}
    if not rougeVariant in m_valuesDict[systemId]:
        m_valuesDict[systemId][rougeVariant] = {}
    if not metric in m_valuesDict[systemId][rougeVariant]:
        m_valuesDict[systemId][rougeVariant][metric] = {}
        
    m_valuesDict[systemId][rougeVariant][metric]['avg'] = value
    
    if not rougeVariant in m_rougeVariants:
        m_rougeVariants.append(rougeVariant)
        
        
if __name__ == '__main__':
    main(ROUGE_SCORES_FILE, OUTPUT_CSV_FILE)