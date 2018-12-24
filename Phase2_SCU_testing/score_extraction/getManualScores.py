import csv

'''
This script reads the Pyramid and Responsiveness scores from the DUC 2005 and 2006 scores files.
It outputs the scores to a CSV file in a format to be used when calaculating correlations in the phase2 post-script.

Provide the input and output files and the column names of the input file.
'''

'''
The original scores file.
2005: e.g. '../../../DUC_data/2005/processed_pans.txt'
2006: e.g. '../../../DUC_data/2006/scoring/2006_modified_scores.txt'
'''
SCORES_FILE = ''
'''
The file to output to.
'''
OUTPUT_CSV_FILE = '' # e.g. '2005ManualScoresAvg.csv'
'''
The column names to use for the input file. Make sure to make a fields:
'setid', 'peerid', 'modifiedscore', 'responsiveness'
2005: ['setid', 'peerid', 'score', 'modifiedscore', 'numSCUs', 'numrepetitions', 'annotatorid', 'pyramidcreator', 'lingqual1', 'lingqual2', 'lingqual3', 'lingqual4', 'lingqual5', 'responsiveness', 'editing']
2006: ['setid', 'peerid', 'modifiedscore', 'numSCUs', 'numrepetitions', 'annotatorid', 'pyramidcreator', 'lingqual1', 'lingqual2', 'lingqual3', 'lingqual4', 'lingqual5', 'responsiveness']
'''
COLUMNS = []


m_valuesDict = {}
m_scoresByEvent = {}

def main(scoresCsvFile, outputCsvFile):
    # read in the scores:
    with open(scoresCsvFile, 'r') as fIn:
        scoresTable = csv.DictReader(fIn, fieldnames=COLUMNS, delimiter='\t')
        for row in scoresTable:
            processRow(row)
    # average the scores:
    averageAll()
    # print out the final scores:
    printStats(outputCsvFile)

def printStats(outputCsvFile):
    with open(outputCsvFile, 'w') as outF:
        outF.write('systemId, pyramid, responsiveness\n')
        for systemId in m_valuesDict:
            outF.write('{}, {}, {}\n'.format(systemId, m_valuesDict[systemId]['pyr'], m_valuesDict[systemId]['res']))
            
        outF.write('\n\n')
        outF.write('eventId, pyramid, responsiveness\n')
        for eventId in m_scoresByEvent:
            outF.write('{}, {}, {}\n'.format(eventId, m_scoresByEvent[eventId]['pyr'], m_scoresByEvent[eventId]['res']))
            
        outF.write('\n\n')
        outF.write('systemId, eventId, pyramid, responsiveness\n')
        for systemId in m_valuesDict:
            for eventId in m_scoresByEvent:
                if eventId in m_valuesDict[systemId]:
                    outF.write('{}, {}, {}, {}\n'.format(systemId, eventId, m_valuesDict[systemId][eventId]['pyr'], m_valuesDict[systemId][eventId]['res']))
                
    
    
def averageAll():
    global m_valuesDict
    global m_scoresByEvent
    
    # get all the scores by systemId and eventId separately:
    for systemId in m_valuesDict:
        allScoresPyramid = [m_valuesDict[systemId][eventId]['pyr'] for eventId in m_valuesDict[systemId]]
        allScoresResponsiveness = [m_valuesDict[systemId][eventId]['res'] for eventId in m_valuesDict[systemId] if m_valuesDict[systemId][eventId]['res'] != -1.0]
        
        for eventId in m_valuesDict[systemId]:
            m_scoresByEvent.setdefault(eventId, {}).setdefault('pyr', []).append(m_valuesDict[systemId][eventId]['pyr'])
            if m_valuesDict[systemId][eventId]['res'] != -1.0:
                m_scoresByEvent[eventId].setdefault('res', []).append(m_valuesDict[systemId][eventId]['res'])
        m_valuesDict[systemId]['pyr'] = reduce(lambda x, y: x + y, allScoresPyramid) / len(allScoresPyramid)
        if len(allScoresResponsiveness) > 0:
            m_valuesDict[systemId]['res'] = reduce(lambda x, y: x + y, allScoresResponsiveness) / len(allScoresResponsiveness)
        else:
            m_valuesDict[systemId]['res'] = -1.0
        
    # average the scores for each event (changes from list to value):
    for eventId in m_scoresByEvent:
        m_scoresByEvent[eventId]['pyr'] = reduce(lambda x, y: x + y, m_scoresByEvent[eventId]['pyr']) / len(m_scoresByEvent[eventId]['pyr'])
        m_scoresByEvent[eventId]['res'] = reduce(lambda x, y: x + y, m_scoresByEvent[eventId]['res']) / len(m_scoresByEvent[eventId]['res'])
    
    
def processRow(row):
    global m_valuesDict
    
    eventId = row['setid']
    try:
        systemId = str(int(row['peerid'])) # sometimes the system is '01' instead of '1'
    except:
        systemId = row['peerid'] # if the peerid is not a number
    pyramidScore = float(row['modifiedscore'])
    #numSCUs = row['numSCUs']
    #numrepetitions = row['numrepetitions']
    
    # The file sometimes contains two data points on a system on an event. Just skip those for now.
    if systemId in m_valuesDict and eventId in m_valuesDict[systemId]:
        return
    
    if row['responsiveness'] != '':
        responsivenessScore = float(row['responsiveness'])
    else:
        responsivenessScore = -1.0
    
    if not systemId in m_valuesDict:
        m_valuesDict[systemId] = {}
    if not eventId in m_valuesDict[systemId]:
        m_valuesDict[systemId][eventId] = {}
    
    # keep the systems scores just processed:
    m_valuesDict[systemId][eventId]['pyr'] = pyramidScore
    m_valuesDict[systemId][eventId]['res'] = responsivenessScore

    
if __name__ == '__main__':
    main(SCORES_FILE, OUTPUT_CSV_FILE)