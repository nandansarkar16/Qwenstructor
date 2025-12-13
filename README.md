# Project repository for CPSC 4710 final project.     
Group members: Howard Dai, Akhil Elangovan, Nandan Sarkar, Ben Xu

# Environment setup
Run ```pip install requirements.txt```    

# Reproducing results 
This returns a table of all model responses across the three versions tested in our paper.             

```python  fitness_calculator.py --data_path /vast/palmer/pi/krishnaswamy_smita/hcd22/GPTeacher/data/val_set.csv --prompt_path fixed_eval_adversarial --eval```

# Running an iteration of AutoDAN 
The model checkpoints are:

- **Original SFT-trained filter:** `nandansarkar/base_qwen3_0-6B_filter`
- **1-epoch adversarially trained model:** `nandansarkar/qwen3_0-6B_adversarial_1`
- **Fully adversarially trained model:** `nandansarkar/qwen3_0-6B_adversarial_final`

Run all commands inside the custom-autodan folder.

1. Genetic forward pass

IF FIRST ITERATION:       
```python  autodan_GA_iter.py --data_path ../data/braingle_Math_annotated.csv --model_path [MODEL PATH] --first```        
This uses the set of initial seed prompts which can be found inside the code itself. 

OTHERWISE ("out_prompts" folder should be populated):       
```python  --data_path ../data/braingle_Math_annotated.csv --prompt_path out_prompts --model_path [MODEL PATH]```      

2. GPT revisions          
This should update the folder called "prompts_to_reword" with several text files. Copy paste the contents of text file into GPT, and paste them into the corresponding numbered text file in "out_prompts". These are the new set of prompts for the next forward pass. 


# Running SFT 

To run supervised fine-tuning (SFT) on the Braingle dataset, first ensure that the dataset is placed in `LLaMA-Factory/data` and that there is a corresponding entry in `LLaMA-Factory/data/dataset_info.json`.

All hyperparameter configurations are stored in the `configs` directory. 

To launch training, update `scripts/train.sh` to point to the desired hyperparameter configuration file, then run:

```bash train.sh```

The training script is compatible with Slurm and other GPU schedulers i.e. ```srun --partition={partition_name} --gres=gpu:{number_of_gpus} bash train.sh```

# Running the CLI for student learning
TODO 
