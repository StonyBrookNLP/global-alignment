## Continual Learning with Global Alignment
This is the repository for the Neurips 24 paper: [Continual Learning with Global Alignment](https://openreview.net/pdf?id=4vp0edVY4o).

### Package Requirement
```
numpy == 1.16.2
torch == 1.9.1
transformers == 3.0.0
```

### Data
Each task data are stored in the ```./data``` directory. For tasks in GLUE benchmark, please download the data in [this link](https://gluebenchmark.com/tasks) to the corresponding sub directories. Other data can be downloaded [here](https://drive.google.com/drive/folders/0Bz8a_Dbh9Qhbfll6bVpmNUtUcFdjYmF2SEpmZUZUcVNiMUw1TWN6RDV3a0JHT3kxLVhVR2M?resourcekey=0-TLwzfR2O-D2aPitmn5o9VQ).

To preprocess the data, please run ```python preprocess.py``` in each sub directory. For Yahoo and DBPedia which need to be split to sub-class tasks, please run ```python preprocess_split.py```.

### Run
The script for running different methods is in  ```./src/script.sh``` (details to be added). 

### Citation
```
@inproceedings{
bai2024continual,
title={Continual Learning with Global Alignment},
author={Xueying Bai and Jinghuan Shang and Yifan Sun and Niranjan Balasubramanian},
booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
year={2024},
url={https://openreview.net/forum?id=4vp0edVY4o}
}
```

