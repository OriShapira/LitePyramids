# LitePyramids
This project includes code, crowdsourcing tasks, and a dataset for manual evaluation of system summaries, via crowdsourcing.
The idea is based on the [Pyramid evaluation method](http://www.aclweb.org/anthology/N04-1019) (Nenkova and Passonneau 2004).

This resource is supplementary to the paper: *TODO: put once published*

## Overview
The overall method consists of two phases:
1. Pyramid creation: Workers extract "summary content units" (SCUs), or sentential facts, from reference summaries (gold annotated summaries on some input text(s) -- topics).
2. System summary evaluation: For each utilized SCU, workers mark if it is contained in a system summary (a summary created automatically by an algorithm). A system summary is then scored as a percentage of SCUs covered. A system's score is the average over its summaries on many topics.

SCUs generated for some topic in the first phase are used across all system summaries of the topic. Therefore, the first phase is done only once, and the SCUs are reused in the second phase.

## Requirments
Checked on Python 2.7, and should work on Python 3.

## The Resource
You can use this resource to:
* Evaluate a new summarization system on the DUC 2005 or 2006 data. This costs $108 on 2006 or $54 on 2005 (only half the topics are available) on Amazon Mechanical Turk.
* Create a new SCU collection for a dataset other than DUC 2005 or 2006, as long as it has reference summary(ies). This costs $1.20 per reference summary on Amazon Mechanical Turk. These new SCUs can then be used to evaluate summaries on the new summarization dataset.
* Use the DUC 2005/2006 SCUs (almost 1500 of them) for any other purpose. Each SCU is a sentence about a fact from a reference summary.
* Reproduce the whole two-phase task of SCU generation and DUC 2005/2006 system evaluations.

### To only evaluate a new system
If you simply want to evaluate your system on the DUC 2005 and 2006 datasets, follow these steps:
1. Get the DUC 2005/2006 datasets from [NIST](https://www-nlpir.nist.gov/projects/duc/data.html).
2. Run your system on the documents of the document sets listed here: Phase1_SCU_writing/dataset/relevant_topics.txt.
3. Keep each output summary in a separate file with the topicId as the name, and put them in a separate folder.
4. Run `Phase2_SCU_testing/processing_scripts/pre_createInputForAMT_newSystem.py <your_system_name> <path_to_summaries_folder> <2005|2006> <path_to_new_output_file_batch1> <path_to_new_output_file_batch2>`
5. In [Amazon Mechanical Turk](https://requester.mturk.com), create a task with the task_properties and task_designLayout in the Phase2_SCU_testing/AMT_task folder.
7. Create a new batch with the output file from step 4.
8. Once the task has finished in AMT, download the results file.
9. Run `python Phase2_SCU_testing/processing_scripts/post_calculateScores_newSystem.py <path_to_AMT_results_file> and the summary scores (per event) and overall system score will be printed out.

Note: Make sure to compare your results with system summaries of the same length. Since this is a recall measure on the SCUs, it would be unfair to compare summaries of different lengths.

### To generate SCUs for a new dataset
If you would like to create SCUs for a new dataset that has reference summaries:
1. Place the reference summaries of the new dataset in a folder, where each file contains the text of the reference summary. The filename should start with the topic ID (or document ID if single-document summary).
2. Run `python Phase1_SCU_writing/processing_scripts/pre_createInputForAMT.py`, after updating the REF_SUMM_FOLDER, EVENT_IDS, OUTPUT_FILE and NUM_REFS_PER_EVENT variables in the script.
3. In [Amazon Mechanical Turk](https://requester.mturk.com), create a task with the task_properties and task_designLayout in the Phase1_SCU_writing/AMT_task folder.
4. Create a new batch with the output file from step 2.
5. Once the task has finished in AMT, download the results file.
6. Create the two SCU files for use in the second phase (you can also choose to have just one, or more than two SCU files):
    1. Run `python Phase1_SCU_writing/processing_scripts/post_selectSCUsFromAMT.py`, after updating the SCUS_RESULTS_CSV, OUTPUT_CSV_PATH and NUM_SCUS_TO_SAMPLE variables in the script.
    2. Run `python Phase1_SCU_writing/processing_scripts/post_useOtherSCUs.py`, after updating the INPUT_QUESTIONS_CSV_PATH, OUTPUT_QUESTIONS_CSV_PATH and NUM_SCUS_PER_REF variables in the script.
7. You may now use the generated SCU files as a resource to evaluate summaries in the second phase.


### Full process

#### DUC_data
You can get the NIST data from [here](https://www-nlpir.nist.gov/projects/duc/data.html).
Only the DUC 2005 and 2006 SCUs generated as part of this research, and the original system scores are available here.
Once you have the full data, put them in the relevant DUC_data folder.

#### Phase1_SCU_writing
1. Run `python Phase1_SCU_writing/processing_scripts/pre_createInputForAMT.py`, after updating the REF_SUMM_FOLDER, EVENT_IDS, OUTPUT_FILE and NUM_REFS_PER_EVENT variables in the script.
2. In [Amazon Mechanical Turk](https://requester.mturk.com), create a task with the task_properties and task_designLayout in the Phase1_SCU_writing/AMT_task folder.
3. Create a new batch with the output file from step 1.
4. Once the task has finished in AMT, download the results file.
5. Create the two SCU files for use in the second phase:
    1. Run `python Phase1_SCU_writing/processing_scripts/post_selectSCUsFromAMT.py`, after updating the SCUS_RESULTS_CSV, OUTPUT_CSV_PATH and NUM_SCUS_TO_SAMPLE variables in the script.
    2. Run `python Phase1_SCU_writing/processing_scripts/post_useOtherSCUs.py`, after updating the INPUT_QUESTIONS_CSV_PATH, OUTPUT_QUESTIONS_CSV_PATH and NUM_SCUS_PER_REF variables in the script.

#### Phase2_SCU_testing
1. Run `python Phase2_SCU_testing/processing_scripts/pre_createInputForAMT.py`, after updating the EVENT_IDS, SYSTEM_IDS, SUMMARIES_FOLDER, OUT_CSV_FILE and QUESTIONS_FILE variables in the script.
2. In [Amazon Mechanical Turk](https://requester.mturk.com), create a task with the task_properties and task_designLayout in the Phase2_SCU_testing/AMT_task folder.
3. Create a new batch with the output file from step 1.
4. Once the task has finished in AMT, download the results file.
5. Run `python Phase2_SCU_testing/processing_scripts/post_calculateScores.py`, after updating the RESULTS_FILE_INPUT, OUTPUT_FILE, MANUAL_SCORES_FILE, and ROUGE_SCORES_FILE variables in the script. You can also pplay around with the configuration variables to see how they change the scores and correlations.

##### Original score extraction
These scripts extract the scores from the DUC data, average the scores, and output them to a format used in step 5 of Phase2 above. These score files are already available.