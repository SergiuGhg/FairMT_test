# FAIRMT Reproduction and Modifications

The files contained here are based on the original FairMT files, however they are modfied for use of different models and steering vectors experiments.

Changes have been made to:
 - add steering vectors to the answer generation code
 - evaluate the answers using Llama Guard 3

## Running the evaluation

The all_answers.py file can be used to obtain the conversation answers using Llama-3.1-8B-Instruct.
The file generate_answers_vectors.py can be modified to use different steering vectors depending on test case.
The evaluate_all.py file will evaluate all answers using the Llama Guard 3 model.

## File description

Along the original files from the FairMT experiment repository one can find the outputs_ files and the evals_ files.
The outputs_ files are the respective outputs for each of the 6 experiments that were run.
The evals_ files are their respective evaluations.

## Findings

The results of the experiment can be found in the PR_findings.pdf file
