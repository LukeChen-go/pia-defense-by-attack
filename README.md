# Defense Against Prompt Injection Attack by Leveraging Attack Techniques

Official code implementation for ACL 2025 accepted paper: Defense Against Prompt Injection Attack by Leveraging Attack Techniques (https://arxiv.org/abs/2411.00459)

<p align="center">
  <img src="figs/attack_intro.jpg" width="650"/>
</p>

### Environment
```
conda creat -n defenseinj python=3.12
conda activate defenseinj
pip install -r requirements.txt

```

### Evaluation

We implement most of the attack and defense methods based on [StruQ](https://github.com/Sizhe-Chen/StruQ).
Special thanks to the original authors for their excellent work!

To evaluate the effectiveness of our defense method based on "Fakecom attack" against "Combined attack" in indirect prompt injection scenario , please use the command below:

```angular2html
python run_evaluation_instruction.py \
--defense injection-completionreal \
 --attack completion_realcmb \
--log_path logs/defense_llama3_inst.txt \ 
--data_path data/crafted_instruction_data_qa.json \ 
--model_path meta-llama/Meta-Llama-3-8B-Instruct

```

To evaluate the effectiveness of our defense method based on "Fakecom-t attack" against "Combined attack" in indirect prompt injection scenario , please use the command below:

```angular2html
python run_evaluation_instruction.py \ 
--defense none --defense_cross_prompt \
--attack completion_realcmb  \
--log_path logs/defense_llama3_inst_cross.txt \ 
--data_path data/crafted_instruction_data_qa.json \ 
--model_path meta-llama/Meta-Llama-3-8B-Instruct
```



