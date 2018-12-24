import os
import random

'''
Creates the input for the MTurk assignment of writing SCUs for summaries.
Provide here the DUC reference summaries folder, the output file, the eventIDs for which to get the summaries, and the number of reference summaries per event.
The output CSV file has the columns: eventId,summId,summary_text
'''

'''
The path to the folder of DUC reference summaries. Each file in the folder is a reference summary,
likely with a name like "D0601.M.250.A.A" (eventId.manual.lengthInWords.assessorID.writerID).
Each file should contain only the text of the summary (can have newlines).
'''
REF_SUMM_FOLDER = '' # e.g. '../../DUC_data/2006/NISTeval/ROUGE/models'
'''
The event IDs for which to create the AMT task. These are the first part of the filenames (like D0601).
example: EVENT_IDS = ['D0628', 'D0629', 'D0643']
'''
EVENT_IDS = []
'''
The path of the file to write the output to. This will be the AMT input file for the SCU writing task.
'''
OUTPUT_FILE = '' # e.g. 'AMT_input.csv'
'''
The number of reference summaries to use for each event.
4 is the commonly used amount in the Pyramid method.
'''
NUM_REFS_PER_EVENT = 4



lines = {}

for filename in os.listdir(REF_SUMM_FOLDER):
    for eventId in EVENT_IDS:
        if not eventId in lines:
            lines[eventId] = []
        fullEventId = eventId #'D' + eventId # NOTE: in 2005, the folder name event IDs are D###
        if filename.startswith(fullEventId):
            # get the text of the reference summary with a proper eventId:
            filepath = os.path.join(REF_SUMM_FOLDER, filename)
            with open(filepath, 'r') as refSumFile:
                summText = refSumFile.read()
            # remove newlines, and replace " with '
            summText = summText.replace('\n', ' ').strip().replace('"', '\'')
            lines[eventId].append('{},{},"{}"\n'.format(eventId, filename, summText))

# choose a sample of ref summs for each event:
allLinesToUse = []
for eventId in EVENT_IDS:
    linesToUse = random.sample(lines[eventId], NUM_REFS_PER_EVENT)
    allLinesToUse.extend(linesToUse)
        
# write the lines out shuffled:
random.shuffle(allLinesToUse)
with open(OUTPUT_FILE, 'w') as outF:
    outF.write('eventId,summId,summary_text\n')
    for line in allLinesToUse:
        outF.write(line)