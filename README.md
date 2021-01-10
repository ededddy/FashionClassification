# Machine learning project at University of Macau

## Training and Testing environment
CPU: Ryzen 1600 6C12T  
RAM: 16 GB RAM @ 3200 MHz   
(CPU Training, GPU is irrelevant)  
1. Python 3.7
2. numpy 1.19.5
3. scikit-learn 0.24.0 (and all its dependencies)

# How to train :
1. Put your data (from UMMoodle) in `./data` folder  
(28 x 28 flattend images and corresponding labels on a different file)  
2. Make sure you have `saved_model` folder created
3. ```python trainer.py``` and wait for it to finish
- The trained model are saved in ```saved_model```
- There is no checkpoint feature

# How to test :
After you trained the models with `trainer.py`,   
make sure there are files start with `Classifer-` in `saved_model` folder.
1. ```python tester.py```
- `tester.py` will load images named `t10k-images-idx3-ubyte.gz`
- `tester.py` will load labels named `t10k-labels-idx3-ubyte.gz`
- Make sure you have above 2 files in `data` folder
Wait for it to run and look at the results 
