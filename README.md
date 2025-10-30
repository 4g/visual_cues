# Do visual cues help vllms ? 

TLDR : Not really. 

## Problem
VLMs are very good at identifying events in videos. But it is hard to focus their attention on a specific action or object, specially if the object or action is a small part of the video. Humans use visual cues to attract attention towards a region or action, e.g. a stop sign or a circle drawn around wrongly spelled text. We experiment if using similar visual cues helps VLMs focus on the right thing. 

![plain video ](assets/plain.gif)![trajectory and bbox cues](assets/trajectory.gif)

## Experiment
### Data

### Cues

### Model

### Results
| model                      | videos      | object_acc | verb_acc | overall_acc |
|---------------------------|-------------|-----------:|---------:|------------:|
| finetuned                  | plain       | 75.80%     | 98.30%   | 85.00%      |
| finetuned                  | trajectory  | 76.21%     | 98.30%   | 85.24%      |
| pretrained | plain       | 35.75%     | 61.30%   | 46.20%      |
| pretrained | trajectory  | 31.74%     | 61.40%   | 43.87%      |


## Reproduce on own

Clone repo

Clone data
`https://huggingface.co/datasets/apurvagup/visual_cues_ssv2/`

### Run training 


