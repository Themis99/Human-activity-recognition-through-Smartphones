# Human Activity Recognition Through Smartphones

In this project, we try to track the physical activities of people through sensors from smartphones placed in different positions of the body. These activities are biking, downstairs, jogging, sitting, standing, upstairs, and walking. **Support Vector Machine** was used for the classification.

## Dataset
The dataset that is used is publicly available from the research of [1]. This dataset includes records from 10 participants who have
mobile phones in different body positions while performing one of the
the following seven activities: 

1. biking
2. downstairs
3. jogging
4. sitting
5. standing
6. upstairs
7. walking

Each of these
participants were equipped with five smartphones in five body positions:

1. One in the right pocket of the jeans
2. One in the left pocket of the jeans
3. One in the belt position towards the right leg using a belt clip
4. One on the right upper arm
5. One on the right wrist

More information about the dataset can be found in the research [1]

## Data cleaning and pre-processing 
The pre-processing of data involved mostly:

1) Selecting the signal from the accelerometer of the right pocket.
The measurements were collected from the sensors of the mobile phone located in the right pocket.

2) Conversion of the three-dimensional signal of the sensor into one
dimension. Each of
of these sensors has three dimensions, namely the x-axis, the
the y-axis and z-axis and their respective values are referred to as these
axes. The orientation of the sensors influences the performance
of the algorithms therefore it is important to try to have
as much as possible a constant orientation to the data which will be
used by the algorithms. Also, the recognition performance
of the sensors can be affected by orientation changes
if the classification algorithms were trained only for a specific
orientation, resulting in a drop in performance. To
minimize these effects instead of using the three
dimensions, the vector length is used which is less
sensitive to orientation changes.  

3) Measure values that are above 1000 correspond to errors in the
sensor. These values are replaced with the previous value of
the sensor.

## Feature extraction
### Initial signal processing - Sampling

During the initial processing of the signal, windows of 20 
seconds are selected with a step of 1 second. This combined with the fact that the
sampling in the dataset has been done at a frequency of 50Hz, means that
each window will have 50*20=1000 samples and the step will be 50 samples.
This leads us to the conclusion that there will be a large overlap between the
different recordings.
A positive aspect of this overlap is that 
we do not lose information in the boundary values of the windows. 

### Feature extraction
After sampling, the extraction of features is followed. The
features we collected for each window are the following:

1) Average of the window values
2) Standard deviation of the window values
3) Distribution asymmetry - skewness
4) Maximum value
5) Minimum value
6) Difference between maximum and minimum value
7) Spectrum power estimation using Welch's method

The type of window in the method of
Welch defined the Hanning window, the length was set to 128 L, and the overlap 50%. For the length of the windows, the following values were tested 64, 128, 256, 512, and 1000. From this experimentation, we found that a length of 128 has the best performance.  

### Principal Component Analysis (PCA)
Because the power spectrum corresponds to many values (in our case 65
values), we chose to experiment with dimensionality reduction techniques. The technique
which we chose is PCA (Principal Component Analysis). Of the total 71 features, after applying the PCA method only 19 features are kept.

### Data normalization   
As mentioned before the SVM (Support Vector Machine) algorithm is used for the classification. This is a distance-based algorithm. Therefore, it is very important that the ranges of values of the features
are the same because otherwise, the algorithm will erroneously consider more
more important a feature whose value range is larger than a feature whose value range is smaller.
Therefore, for this project, we chose linear normalization as a method to normalize our data. This method transforms all
the data in the value range [0,1]. Another advantage of this method is the avoidance of
numerical difficulties in the calculation. Because the values of the kernel
usually depend on the inner products of the characteristic vectors
(feature vectors), e.g. the linear and polynomial kernel, the large feature values
 may cause numerical problems. Also, it is
very important to pay attention to how we normalize the train and
the test set to avoid introducing bias into our model. Thus, the train set is firstly normalized and then the same parameters are used to normalize the test set

## Evaluation
As mentioned before, the algorithm used for the classification is SVM. We also have data from ten participants of the experiment.
Leave One Subject Out (LOSO) was chosen as the evaluation method
which is a slight variation of Leave One Out (LOO). With this method we train the SVM algorithm by selecting as training data nine of the ten participants leaving one participant for evaluation. The participant to use for evaluation is selected randomly and the procedure is repeated 10 times until all participants are used for evaluation.

### Results
The overall accuracy is 82.16%
Precision, Recall and F1-Score are calculated for each class: 

| Class  | Presicion | Recall | F1-Score |
| ------- | ----------  | ----- | ----- |
|  walking   | 0.97 | 0.97 | 0.97 |
| standing | 0.88 | 0.91 | 0.90 |
| jogging | 0.98 | 0.99 | 0.99 |
| sitting | 0.51 | 0.93 | 0.66 |
| biking | 0.65 | 0.08 | 0.14 |
| upstairs | 0.96 | 0.97 | 0.96 |
| downstairs | 0.92 | 0.89 | 0.91 |

| Statistic  | Presicion | Recall | F1-Score |
| ------- | ----------  | ----- | ----- |
|  micro avg   | 0.82 | 0.82 | 0.82 |
| macro avg | 0.84 | 0.82 | 0.79 |
| weighted | 0.84 | 0.82 | 0.79 |

Finally the confusion matrix is presended:

![Confmat](https://user-images.githubusercontent.com/46052843/189058420-4d720d76-6b58-4431-b061-f56cd183a6ad.png)

## Discussion
The algorithm confuses walking with downstairs in a small percentage of 10%.
Then, it confuses standing with sitting at a rate of 89.7%. The next
activity is jogging where the confusion here is negligible, he 
predictics correctly at 99.5%. The next activity is sitting which 
confuses a bit with the other activities but in very small percentages. Similarly
for biking and upstars. Finally, there is a confusion of downstairs with
walking at a rate of 6.7%.
In summary there is a strong confusion of standing
and sitting activities. However, the remaining classes are separated to a very good degree with
jogging being identified at a rate of 99%. Also, the overall accuracy achieved by the
algorithm, 82.16% is quite satisfactory for the recognition of human
activity.

## References
[1] Shoaib, Muhammad & Bosch, Stephan & Incel, Ozlem & Scholten, Hans & Havinga,
Paul. (2014). Fusion of Smartphone Motion Sensors for Physical Activity Recognition.
Sensors (Basel, Switzerland). 14. 10146-10176. 10.3390/s140610146.
