import os
from random import shuffle
import csv

'''
Creates the input for the MTurk assignment of testing system summaries.
Provide here the DUC system summaries folder, the output file, the SCUs file (from phase 1), the eventIDs for which to get the summaries, and the number of reference summaries per event, and the system IDs to be evaluted.
The output CSV file has the columns: 
eventId,summaryId,qIdList,summary_text,statement_1,statement_2,statement_3,statement_4,statement_5,statement_6,statement_7,statement_8,statement_9,statement_10,statement_11,statement_12,statement_13,statement_14,statement_15,statement_16

Since two different batches of 16 SCUs need to be run, create two different AMT input files for this phase,
one for each SCU batch file from phase 1.
'''

'''
The event IDs for which to crowdsource scores.
'''
EVENT_IDS = [] # e.g. ['D0629', 'D0650', 'D0647']
'''
The system IDs for which to crowdsource scores.
'''
SYSTEM_IDS = [] # e.g. ['28', '6', '33', '14']
'''
The path to the folder of DUC system summaries. Each file in the folder is a system summary,
likely with a name like "D0601.M.250.A.16" (eventId.manual.lengthInWords.assessorID.systemID).
Each file should contain only the text of the summary (can have newlines).
'''
SUMMARIES_FOLDER = '' # e.g. '../../DUC_data/2006/NISTeval2/ROUGE/peers'
'''
The path of the file to write the output to. This will be the AMT input file for the system summary evaluation task.
'''
OUT_CSV_FILE = '' # e.g. 'AMT_input.csv'
'''
The path of the SCUs file from which the testing SCUs will be taken.
This is the dataset file created in the SCU writing phase with fields:
eventId, questionId, questionText, answer, author, sourceSummaryId, forUse
'''
QUESTIONS_FILE = '' # e.g. '../../Phase1_SCU_writing/processing_scripts/SCUs_batch1.csv'


# notice that the eventIDs may not have the 'D' prefix, and may need to be added in the questions file eventId column

# get the SCUs to use for all events:
eventQuestions = {} # { eventId -> { scuID -> SCUtext } }
eventQuestionsLists = {} # { eventId -> [ scuIDs ] }
with open(QUESTIONS_FILE, 'r') as inF:
    csv_reader = csv.DictReader(inF)
    for row in csv_reader:
        if row['forUse'] == '1':
            eventId = row['eventId']
            qId = row['questionId']
            qText = row['questionText']
            eventQuestions.setdefault(eventId, {})[qId] = qText
            eventQuestionsLists.setdefault(eventId, []).append(qId)


csvOutputLines = []
summFilenames = os.listdir(SUMMARIES_FOLDER)
for fn in summFilenames:
    found = False
    # for the relevant eventIDs and systemIDs, get the texts and SCUs to put in the output file:
    for eventId in EVENT_IDS:
        for systemId in SYSTEM_IDS:
            if fn.startswith(eventId + '.') and fn.endswith('.' + systemId):
                with open(os.path.join(SUMMARIES_FOLDER, fn), 'r') as fIn:
                    text = fIn.read()
                # replace problematic quotation characters:
                summText = text.replace('\n', ' ').replace('``', '\'\'').replace('"', '\'\'').strip()#.replace(',', '&#44').strip()#'&#44').strip()
                
                questionsText = ','.join(['"{}"'.format(eventQuestions[eventId][qId].replace('``', '\'\'').replace('"', '\'\'').strip()) for qId in eventQuestionsLists[eventId]])
                newLine = '{},{},"{}","{}",{}\n'.format(eventId, fn, str(eventQuestionsLists[eventId]), summText, questionsText)
                csvOutputLines.append(newLine)
                
                found = True
                break
        if found:
            break
           
# write out the output:
shuffle(csvOutputLines)
with open(OUT_CSV_FILE, 'w') as fOut:
    fOut.write('eventId,summaryId,qIdList,summary_text,statement_1,statement_2,statement_3,statement_4,statement_5,statement_6,statement_7,statement_8,statement_9,statement_10,statement_11,statement_12,statement_13,statement_14,statement_15,statement_16\n')
    for outLine in csvOutputLines:
        fOut.write(outLine)
        
        
for eventId in eventQuestions:
    print(eventId)
    for qId in eventQuestions[eventId]:
        print(eventQuestions[eventId][qId])