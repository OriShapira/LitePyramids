import csv
import random

'''
This script creates a SCUs file based on another one, but samples 16 different questions for each event.
The purspose is to create two 16-SCU batches in order to evaluate system summaries (in phase 2) on 32 questions (two bacthes of 16).
'''

'''
The SCUs CSV for which to create a separate one with different chosen SCUs for the system summary evaluation phase.
The fields are:
    eventId,questionId,questionText,answer,author,sourceSummaryId,forUse
'''
INPUT_QUESTIONS_CSV_PATH = '' # e.g. 'SCUs_batch1.csv'
'''
The SCUs CSV to create with different chosen SCUs for the system summary evaluation phase.
The fields are:
    eventId,questionId,questionText,answer,author,sourceSummaryId,forUse
'''
OUTPUT_QUESTIONS_CSV_PATH = '' # e.g. 'SCUs_batch2.csv'
'''
The max number of SCUs to use for each reference summary when testing in the next phase (these will be marked in the dataset).
(There may be less available.)
'''
NUM_SCUS_PER_REF = 4


def main(inputQuestionsFile, outputQuestionsFile):
    # the question IDs not used in the first CSV
    allQuestionsNotUsed = {} # { summId -> [questionIds] }
    # the question IDs already used in the first CSV
    allQuestionsUsed = {} # { summId -> [questionIds] }
    
    # read in the first CSV:
    with open(inputQuestionsFile, 'r') as inF:
        csv_reader = csv.DictReader(inF)
        for row in csv_reader:
            summId = row['sourceSummaryId']
            forUse = row['forUse']
            questionId = row['questionId']
            if forUse == '1':
                allQuestionsUsed.setdefault(summId, []).append(questionId)
            else:
                allQuestionsNotUsed.setdefault(summId, []).append(questionId)

    # For each summary ID, choose 4 SCUs (question IDs) that were not used in the first CSV
    # Notice that if there aren't enough SCUs for this summId, you will be notified in the command line.
    # In this case, manually choose other SCUs from other reference summaries of this event (and mark the forUse field as 1).
    newQuestionsToUse = {}
    for summId in allQuestionsNotUsed:
        if len(allQuestionsNotUsed[summId]) >= NUM_SCUS_PER_REF:
            qIdsToUseNew = random.sample(allQuestionsNotUsed[summId], NUM_SCUS_PER_REF)
        else:
            qIdsToUseNew = allQuestionsNotUsed[summId]
            print('WARNING: Summary ' + summId + ' is missing ' + str(NUM_SCUS_PER_REF - len(allQuestionsNotUsed[summId])) + ' SCUs!!!')
        newQuestionsToUse[summId] = qIdsToUseNew
        
    # write out the new SCUs table to the output file:
    lines = ['eventId,questionId,questionText,answer,author,sourceSummaryId,forUse\n']
    with open(inputQuestionsFile, 'r') as inF:
        csv_reader = csv.DictReader(inF)
        for row in csv_reader:
            eventId = row['eventId']
            questionId = row['questionId']
            questionText = row['questionText']
            answer = row['answer']
            author = row['author']
            sourceSummaryId = row['sourceSummaryId']
            
            if questionId in newQuestionsToUse[sourceSummaryId]:
                lines.append('{},{},"{}",{},{},{},1\n'.format(eventId, questionId, questionText, answer, author, sourceSummaryId))
            else:
                lines.append('{},{},"{}",{},{},{},0\n'.format(eventId, questionId, questionText, answer, author, sourceSummaryId))
        
    with open(outputQuestionsFile, 'w') as outF:
        for line in lines:
            outF.write(line)
    
if __name__ == '__main__':
    main(INPUT_QUESTIONS_CSV_PATH, OUTPUT_QUESTIONS_CSV_PATH)