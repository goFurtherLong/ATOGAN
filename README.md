# ATOGAN



## Before you run the code
  - the code is based on [MUSE] (https://github.com/facebookresearch/MUSE), to learn more basic details please read MUSE.
  - please modify the path of test set in file .src.evaluation.word_translation.py Line 17.

To run our model, please read and run three .bat files in order.

## shell-1-adv-training.bat
  Here we train the adversarial model. Notice, you need to modify the *emb_path_prefix* where stores the dataset (eg, MUSE dataset).
  
  Since our model is bi-directional, you can get both s2t and t2s mappings just by modifying the *other_language* in line 6 for once.
  (eg, da. and we get mapping.pth for en2da and mapping2.pth for da2en in one run.)
  
  The results will be stored in /dump/muse-adv/en-da

## shell-2-refine.bat
  Here we refine the  mappings learned from adversarial model. Same modification should be done as in shell-1.
  
  Notice that, we seperately refine the s2t and t2s in this procedure, and run twice to get the refined mappings for two directions.
  (we get mapping.pth for en-da, and get mapping.pth for da-en in two dirrerent experiments.)
  
  The results will be stored in /dump/muse-refine/en-da and /dump/muse-refine/da-en
 
 ## shell-3-local-mapping.bat
  Here we implement the Non-linear Local Mapping Strategy. Same modification should be done as in shell-1.
  
  We also seperately do this strategy for two directions. And get the reuslts in in two dirrerent experiments.
  
  The results will be stored in /dump/muse-localMapping/en-da and /dump/muse-localMapping/da-en, respectively.
 
  
  
  
  
