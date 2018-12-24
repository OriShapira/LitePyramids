import csv

'''
This script creates files with scores of specific systems/events (out of a *ManualScoresAvg.csv file).
Provide the input and output files, the system IDs and the event IDs to use.
'''

'''
The system IDs for which to compute the average scores.
'''
SYSTEM_IDS = [] # e.g. ['23', '28', '6', '33', '14', '25', '1']
'''
The event IDs for which to compute the average scores.
'''
EVENT_IDS = [] # e.g. ['D0603', 'D0605', 'D0608', 'D0614', 'D0630', 'D0631', 'D0640']
'''
The input scores file.
'''
INPUT_FILE = '' # e.g. '2006ManualScoresAvg.csv'
'''
The output scores file.
'''
OUTPUT_FILE = '' # e.g. '2006ManualScoresPartial.csv'



# read in the individual system summary scores:
systemScores = {sysId:[] for sysId in SYSTEM_IDS}
with open(INPUT_FILE, 'r') as fIn:
    startRead = False
    for row in fIn:
        row = row.strip()
        if not startRead:
            if row == 'systemId, eventId, pyramid, responsiveness':
                startRead = True
        elif row != '':
            parts = row.split(',')
            systemId = parts[0].strip()
            eventId = parts[1].strip()
            pyrScore = float(parts[2].strip())
            
            if systemId in SYSTEM_IDS and eventId in EVENT_IDS:
                systemScores[systemId].append(pyrScore)

#print(systemScores)
          
# average the scores for the systems:          
finalSysScores = {sysId : reduce(lambda x, y: x + y, systemScores[sysId]) / len(systemScores[sysId]) for sysId in SYSTEM_IDS}
# output to file:
with open(OUTPUT_FILE, 'w') as fOut:
    fOut.write('Events: {}\n\n'.format(';'.join(EVENT_IDS)))
    fOut.write('systemId, pyramid\n')
    for sysId in SYSTEM_IDS:
        fOut.write('{}, {}\n'.format(sysId, finalSysScores[sysId]))