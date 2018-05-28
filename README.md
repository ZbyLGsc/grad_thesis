#  Defect Detection Based On Transfer Learning and Model Ensemble

## 1.Introduction

 This project contains all the research code of wood floor defect detection. We employ transfer learning, in which 
 we retrain the Google *Inception v3* and Google *Mobilenet* for our problem. These retrained models are further ensembled to achieve lower *false negative rate*. Besides, we provide code of algorithm evaluation, interface of our defect detection algorithm, example graphical user interface and dataset generation.

The structure of this project can be unfolded as follows:

- image_retraining: retraining the Google *Inception* and *Mobilenet* using wood floor datasets.

- trained_model: trained parameter files of the models. 

- interface: API of our algorithm, which can be used in other application.

- gui: some example graphical user interface of wood floor defect detection, in which our algorithm is used. 

- evaluation: code for evaluating our algorithm(such as accuracy, false negative rate...).

- preprocess: code for generating wood floor datasets

- util: some useful tools, such as image format conversion.

**Authors:** Boyu Zhou, Xin He and Zhongyi Zhou, all from [School of Mechanical Engineering, Shanghai Jiao Tong University](http://me.sjtu.edu.cn/).

## 2.Prerequisities
  Our testing environment: **Ubuntu** 16.04.

  We use **Tensorflow** to implement main part of our algorithm. 
  
  **OpenCV** is used to do some image pre-processing.

  **PyQt** is also used for creating graphical user interface(GUI). 

  All code are written in python, so no compilation is needed. All script can be run immediately.

  For more information, refer to:

  [**Tensorflow**](https://tensorflow.google.cn/)

  [**OpenCV**](https://opencv.org/)

  [**PyQt**](https://wiki.python.org/moin/PyQt4)

## 3.Train your models using transfer learning

Using transfer learning on your own datasets is simple and straight forward. 

First, you should arrange your data all in one folder, which contains several subfolder. Each subfolder should 
only contains one category of data.

For example, now you have a wood floor defect dataset, in which there are four type of defects:
dot, cut, abrasion and spot. Then there should be some folders like:

```
/home/user_name/data/defect/dot    
/home/user_name/data/defect/cut
/home/user_name/data/defect/abrasion
/home/user_name/data/defect/spot
```

After you put your data in the right place, run:
```
cd image_retraining/
python retrain.py --image_dir /home/user_name/data/defect
```

The script has many other options. You can get a full listing with:
```
python retrain.py -h
```

Once the script is run, it read your data and start to retrain the model. After finishing the retrained model
will be saved as a .pb file and .txt file. By default they are saved as */tmp/output_graph.pb* and */tmp/output_labels.txt*, which can be changed as you wish.


## 4.Use your trained models

We already provide interface for the trained models. 

To use your trained model, put the *.pb* and *.txt* files generated in the last step into the *trained_model* folder. Remember that the file should be in accordance with its type(Inception, mobilenet).

After you put the two file at the right place, script in *interface* will use them for prediction. Here is how you should use *interface*

  <div align=center>
  <img src="https://github.com/ZbyLGsc/grad_thesis/pictures/1.png" width = "400" height = "200">
  </div>

For more information about how to use *model_interface.py* and *ensemble_interface.py*, see *evaluation* and
*gui*, in which we use these interfaces.  

## 5.Examples of graphical user interface

run:
```
cd gui/
python qt_gui2.py
```

Then you should press **Open** and select a image. A example image *029(3).tif* is provided.

If the everything run correctly, you should see something like this:
  <div align=center>
  <img src="https://github.com/ZbyLGsc/grad_thesis/pictures/2.png" width = "200" height = "400">
  </div>

## 6.Simulated dataset generation


## 7. Useful tools


## 8.Acknowledgements
  We use **** for () and .


## 9.Licence
The source code is released under [GPLv3](http://www.gnu.org/licenses/) license.



