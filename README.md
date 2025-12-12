# Project repository for CPSC 4710 final project.     
Group members: Howard Dai, Akhil Elangovan, Nandan Sarkar, Ben Xu

# Environment setup
Run ```pip install requirements.txt```    

# Reproducing results 
This returns a table of all model responses across the three versions tested in our paper.             

```python  fitness_calculator.py --data_path /vast/palmer/pi/krishnaswamy_smita/hcd22/GPTeacher/data/val_set.csv --prompt_path fixed_eval_adversarial --eval```

# Running an iteration of AutoDAN 
Paths to the original SFT-trained filter, 1-epoch adversarially trained, and fully adversarially trained, are nandansarkar/base_qwen3_0-6B_filter, nandansarkar/qwen3_0-6B_adversarial_1, nandansarkar/qwen3_0-6B_adversarial_final, respectively. 

Run all commands inside the custom-autodan folder.

1. Genetic forward pass

IF FIRST ITERATION:       
```python  autodan_GA_iter.py --data_path ../data/braingle_Math_annotated.csv --model_path [MODEL PATH] --first```        
This uses the set of initial seed prompts which can be found inside the code itself. 

OTHERWISE ("out_prompts" folder should be populated):       
```python  --data_path ../data/braingle_Math_annotated.csv --prompt_path out_prompts --model_path [MODEL PATH]```      

2. GPT revisions          
This should update the folder called "prompts_to_reword" with several text files. Copy paste the contents of text file into GPT, and paste them into the corresponding numbered text file in "out_prompts". These are the new set of prompts for the next forward pass. 

