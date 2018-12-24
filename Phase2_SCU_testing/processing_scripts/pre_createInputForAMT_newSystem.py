import os
from random import shuffle
import csv
import sys

'''
Creates the input for the MTurk assignment of testing a system summary.

Make sure you have a folder with your system summaries. Each file in the folder contains a system summary, and the name of the file is the event ID.
Usage: pre_createInputForAMT_newSystem.py <your_system_name> <path_to_summaries_folder> <2005|2006> <path_to_new_output_file_batch1> <path_to_new_output_file_batch2>
'''

try:
    SYSTEM_NAME = sys.argv[1]
    SUMMARIES_FOLDER = sys.argv[2]
    DUC_YEAR = sys.argv[3]
    OUT_CSV_FILES = [sys.argv[4], sys.argv[5]]
except:
    print('Usage: pre_createInputForAMT_newSystem.py <system_name> <path_to_summaries_folder> <2005|2006> <path_to_new_output_file_batch1> <path_to_new_output_file_batch2>')
    sys.exit()

if DUC_YEAR == '2005':
    QUESTIONS_FILES = ['Phase1_SCU_writing/dataset/DUC2005/batch1.csv', 'Phase1_SCU_writing/dataset/DUC2005/batch2.csv']
elif DUC_YEAR == '2006':
    QUESTIONS_FILES = ['Phase1_SCU_writing/dataset/DUC2006/batch1.csv', 'Phase1_SCU_writing/dataset/DUC2006/batch2.csv']
else:
    print('Second argument must be either 2005 or 2006.')
    sys.exit()


# notice that the eventIDs may not have the 'D' prefix, and may need to be added in the questions file eventId column

# get the SCUs to use for all events:
for batchNum, questionFile in enumerate(QUESTIONS_FILES):
    eventQuestions = {} # { eventId -> { scuID -> SCUtext } }
    eventQuestionsLists = {} # { eventId -> [ scuIDs ] }
    with open(questionFile, 'r') as inF:
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
        # the filename should be the event ID, and it should have SCUs prepared for it:
        eventId = fn
        if fn not in eventQuestions:
            print('The event {} was not found in the SCUs resource'.format(eventId))
            continue
        
        # get the texts and SCUs to put in the output file:
        with open(os.path.join(SUMMARIES_FOLDER, fn), 'r') as fIn:
            text = fIn.read()
        # replace problematic quotation characters:
        summText = text.replace('\n', ' ').replace('``', '\'\'').replace('"', '\'\'').strip()#.replace(',', '&#44').strip()#'&#44').strip()
        
        questionsText = ','.join(['"{}"'.format(eventQuestions[eventId][qId].replace('``', '\'\'').replace('"', '\'\'').strip()) for qId in eventQuestionsLists[eventId]])
        newLine = '{},{},"{}","{}",{}\n'.format(eventId, SYSTEM_NAME, str(eventQuestionsLists[eventId]), summText, questionsText)
        csvOutputLines.append(newLine)
               
    # write out the output:
    shuffle(csvOutputLines)
    with open(OUT_CSV_FILES[batchNum], 'w') as fOut:
        fOut.write('eventId,summaryId,qIdList,summary_text,statement_1,statement_2,statement_3,statement_4,statement_5,statement_6,statement_7,statement_8,statement_9,statement_10,statement_11,statement_12,statement_13,statement_14,statement_15,statement_16\n')
        for outLine in csvOutputLines:
            fOut.write(outLine)
            
    #for eventId in eventQuestions:
    #    print(eventId)
    #    for qId in eventQuestions[eventId]:
    #        print(eventQuestions[eventId][qId])