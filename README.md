# ONR
Tensorflow CNN on MNIST dataset with B/S interface for handwrite number recognition

## Requirements

* [MNIST dataset](http://yann.lecun.com/exdb/mnist/)
* numpy==1.13.3
* tensorflow==1.2.1
* (If you have tensorflow-gpu, then cuDNN and CUDA are needed. Be careful of version conflict:))
* os
* time
* json
* http
* struct

  Use the requirements.txt by typing this at your terminal:
  ```
  pip install -r requirements.txt
  ```

## How to use?
> 1. Download [MNIST dataset](http://yann.lecun.com/exdb/mnist/) (4 files)
> 2. Unzip all 4 files and put them into `MNIST_Data` folder (parallel to this README) in the project.
> 3. Run `server.py`.
> * (If it is the first time for running, training the model will take less than ten minutes.)
> 4. After `Server Ready` is printed, open the browser and go to `localhost:8000`.
> 5. Draw you digit (from 0 to 9) in the canvas, and press `recognize`.
> 6. Wait for the result to show.
> * Might not give you the right answer, especially when you draw number 6:)
> * Have fun!

## Model detail
### Tensorboard for training
![image](https://github.com/blahBlahhh/ONR/blob/master/ReadMeImg/convnet_graph.png)
### Tensorboard for predicting
![image](https://github.com/blahBlahhh/ONR/blob/master/ReadMeImg/predict_graph.png)
