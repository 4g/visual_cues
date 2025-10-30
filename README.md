# Do visual cues help vllms ? 

TLDR : Not really. 

## Problem
Identify objects and actions in a video. It seems intuitive that drawing bounding boxes around the objects in question should make it easier for vlm to focus on them and increase its accuracy in identifying correct verb and objects. 

![plain video ](assets/plain.gif)![trajectory and bbox cues](assets/trajectory.gif)



## Results
| model                      | videos      | object_acc | verb_acc | overall_acc |
|---------------------------|-------------|-----------:|---------:|------------:|
| finetuned                  | plain       | 75.80%     | 98.30%   | 85.00%      |
| finetuned                  | trajectory  | 76.21%     | 98.30%   | 85.24%      |
| pretrained | plain       | 35.75%     | 61.30%   | 46.20%      |
| pretrained | trajectory  | 31.74%     | 61.40%   | 43.87%      |
