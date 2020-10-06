# Query-Key Normalization for Transformers
Alex Henry, Chief AI Officer, Cyndx

This repo contains code for reproducing the results in our paper "Query-Key Normalization for Transformers" (publication forthcoming in Findings of EMNLP 2020).  Our paper builds off the research in "Transformers without Tears" (Nguyen and Salazar, 2019) and accordingly the code for our paper builds off the code they released for theirs (See https://github.com/tnq177/transformers_without_tears).

To reproduce our results on en-vi, unzip the data.zip folder in transformers_without_tears_head_normalized_attention and run the commands in fastBPE_commands.txt.  Then run the code with the input in bash_input.txt.  Our results for other language pairs can be reproduced using the settings and instructions in our paper.

We've also included output from the Nguyen and Salazar model and from ours in the example_outputs folder, which can be scored using score_test_data.ipynb.

Lastly the simple_demo folder contains The Annotated Transformer notebook with and without QKNorm to illustrate the idea in a setting that's easy to interact with.  This notebook is the source of the heatmaps in Figure 1 and Figure 2 of our paper.
