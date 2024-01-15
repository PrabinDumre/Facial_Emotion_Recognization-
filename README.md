# Facial_Emotion


Bhava: An analysis of Emotion Based On Artificial Intelligence
Shashikala H K , Arastu Thakur, Prabin Dumre, Somnath Bhattarai, Muskan Gupta, Priyanka Shah

Department of Computer Science and Engineering,
Faculty of Engineering and Technology, JAIN (Deemed-to-be) University
Bangalore, India
hk.shashikala@jainuniversity.ac.in; 21BTRCS239@jainuniversity.ac.in; 21BTRCS266@jainuniversity.ac.in; 21BTRCS271@jainuniversity.ac.in;
21BTRCS258@jainuniversity.ac.in; 21BTRCS267@jainuniversity.ac.in

 
Abstract- In this paper the major focus is on Facial Emotion Recognition and How it is resembles our Emotions. Emotions are viable part of a human as every thought or action of a human can be defined as his/her emotion which they depict and by analyzing it we can get to know a lot about one individual. The facial emotion recognition gives in detail account of a person’s behavior and their thoughts. Our research majorly focuses on recognizing a person’s emotion with respect to the analysis of their face and also in this paper we have conducted various phases of preprocessing and dealt with the change in accuracy with respect to it. In our model the major feature is that we are using the combination of extraction of features through the Local Binary Pattern (LBP) along with the help of region based Oriented Fast and rotated Brief (ORB) also we have used this in terms of Convolutional Neural Network (CNN). Our mechanism is working on the model of classification that is purely based on segmented layers of CNN which we have used to classify emotions in the terms of mental and emotional to study in depth. Our model majorly performs its routine action totally based on its segmented layers that is four layers of CNN along with two feature classifiers.

Keywords-FER, Neural Network, Naïve Bayes, DL, ML, NLP, Feature Extraction, Facial Expression.

	INTRODUCTION
The face of a human and emotions on it plays a great role in non-verbal communication. Facial emotion is one of crucial player in the field of oral communication which turns a communication in way better and much more potent. It is also the one that is key identifier in etiquette, frame of mind, psyche , malfeasance, etc.,.The model is inattentive to the factors that are not primarily required i.e., gender, origin, ethnicity, racial complexity and face complexity. The approach that we are going to use is 
capable enough to find the emotion and perform analysis over them regardless the mentioned factor. The initial stage of Facial Emotion Recognition is beginning with detecting the face which may exist either as collection of images or a video or maybe an individual image. The image may directly contain a face or maybe it can be of image with some kind of background. The major thing that is to be considered during the face detection is separation of images from the background which i.e., taking out the face from the collection. The detection of face plays major role in the field of research along with security purpose, it can be used in the feature recognition of certain face or it can also be used in automation system of cameras. The major domain that we can use includes the area of recognition and indexing. 

 Figure 1. Simple Block Based View of Facial Emotion Recognition

In this research the major thing that we have done is model creation for Facial Emotion Recognition with the help of CNN. The approach that we are going to use is capable enough of performing tasks at real time which i.e., means it is dynamic and can get it’s input from the webcam and then work over it to generate the expected result.
The major consideration of the proposed method are:
	CNN method for recognizing facial features with extraction carried out with LBP, ORB and CNN.
	The proposed method has four-layer ‘Conv-Net’ for better recognition.
	Indulging different datasets to increase the detection accuracy.
 
 

	  Related Work and Methodology
Related Work
Several studies in the field of facial emotion recognition have been carried out since many years. After pandemic of Corona the increase in research of Facial Emotion Recognition was seen. As we know for any of the experiment there is some loophole and there is a room for research The objective of this research is to create a new system which gives better accuracy for the pre-existing datasets and analyzes them in efficient manner. In existing methodologies some lack in identifying grayscale images while some has feature lack in pixel-based evaluation. In [1], Compound approach of facial emotion detection consisting of two- stage of recognition, the author of the paper has mentioned about how one stage and two stage model can enhance the data recognition in the meantime. Similar to the context in [2], a combined approach of CNN and LSTM has been used to improvise the accuracy and extract facial expression. [3], In the paper as mentioned the feature vector extraction is useful in extraction through which the error is reduced and the mechanism supports the range of data to be improvised. 
As of experiment with some advantages there exists some deficiency. In the field of facial emotion recognition, the major hindrance that we came across and that drawn our attention includes the accuracy of certain model, their learning capability and approach of machine learning, there in detail exploration of machine learning, the factor of lacking in recognizing actions or behavior’s. Also, majority of the model do lack the performance over different datasets. After referring to the articles and journals the final result is that there is need of impact identification technology which makes the perfect use of the dataset in providing accuracy in results.
Paper Name	Author	Objective	Conclusion
Multi-Modal Emotion Recognition from Speech and Facial Expression Based on Deep Learning	Linqin Cai; Jiangong Dong; Min Wei	Multi-scale Kernel Convolutional Block to increase accuracy based on images and speech.	It used CNN along with LSTM to provide accuracy and detect features.
Feature Vector Extraction Technique for Facial
Emotion Recognition Using Facial Landmarks 	Alwin Poulose, Jung Hwan Kim and Dong Seog Han 	Reducing classification error and to achieve 99.96% accuracy in classification.	A vector extraction technique that combines the facial image pixel values with facial landmarks.
Two-stage Recognition and Beyond for Compound Facial Emotion Recognition	Zhiyuan Zhang; Miao Yi; Juan Xu; Rong Zhang; JianPing Shen	Symmetrical emotion labels enhancing and robust results.	Compound emotion recognition technique with elimination of noisy data.
Facial emotion recognition based on LDA and Facial
Landmark Detection 	Lanxin Sun; JunBo Dai; Xunbing Shen	Use of dimension reduction technique of supervised learning and minimizing intra class variance.	The comparative analysis of accuracy of emotion recognition in which facial landmark gave best possible result.
Emotion Detection with Facial Feature Recognition Using CNN & OpenCV	Sarwesh Giri; Gurchetan Singh; Babul Kumar; Mehakpreet Singh; Deepanker Vashisht	Human- computer domain interaction (HCI) based research and demonstration of the emotion detection.	This approach was constructed using comprehensive research based on CNN algorithm with conjunction of Keras, TensorFlow and retraining principles.
Table 1: Analysis of Previous works.

Proposed Research Methodology
When we are talking about machine learning the thing we should start with is deep neural network and the major key player in the field that is Convolutional neural network which is one of the form of deep neural network that has it usage in processing of vision based images and also in machine vision i.e., computer vision. With a lot of features some deficiency also exists the major deficiency that we found during our research the contradiction in cases of gleaming contradiction along with location-based contradiction which exists mostly in the CNN. In contrast to the methods that we are having in the meantime our method stands out with a structure that is segmented and is arranged in sequential fashion. Our approach performs the task in segments and they all are in sequenced manner i.e., beginning with modelling of the data after that analysis there after prediction based on the data and at last producing a final output which is performed and have been through these all segments.
The major distinctions of our proposed methodology are: 
	Use of LBP in extracting the features.
	Differences in the field of Mathematical Modelling.
	In depth architecture with layered approach.
	LBP (Local binary Pattern)
Local Binary pattern (LBP) is the word that is used in defining the properties of picture on local range or level which means it defines the properties of picture which include appearance features of an image. The advantage that is associated with LBP is the rotating factor and deviation in case of gray scaling. The LBP is one of the major classifying factors that we are considering in the deployment of this machine learning model. As it is simple to use and can serve for long term it is one of the most common techniques that is been used in detection of objects and expressions of faces for extracting required features. The initialization for the LBP begins with the pixel value which is created as a 8 bit binary number following 3*3 neighboring range of the pixel and it does for each pixel. Later on after getting all into segments by binary operator it is turned to the corresponding decimal where the LBP gets result from binary based image patches. The equation that we have considered for calculation for the LBP in our model is: 
(xc, yc):
LBPP, R (Xc, Yc) =∑_(n=0)^7▒2^n S(in - ic) ,S(x) = {_(0,x<0)^(1,x≥0)
where, ic=gray value of center pixel,
in=gray value of neighboring pixel of ic,
P=8 maximum of 8 neighbors of enter pixel.             
                                 R = 1 for selected box of 3 × 3
We are obtaining the value of LBP in the scenario after tracking in clockwise radical the available bins in the radical. Now the next approach is based on the classification of the input from the code that we generated from LBP and it is the one which is going to serve as input for the next. The LBP provides the features and works efficiently even when there is some hindrance in the luminous or maybe it can exist in the scenario of pixels that means the relativity of pixels. With these all advantages and after considering this we concluded that it is ideal suit when we are working for real-time detection. 
In the below figure here we have used (P, R) that is to represent the neighbors of P which are the sample points that are considered in the radical which has an Radius R and in this we can clearly observe that there is even spacing. In this there is symmetric arrangement inside the circle that we can also observe. After considering LBP we have considered the LBP following uniform pattern that means that either it can be 0 – 1 or exist between 1-0 which makes the whole pattern uniform in case of a string that constitutes of circular bits. Considering example: 000000, 001100, 111000 10101100 we can see these all are patterns which are uniform.
 
           Figure 2: Three samples of extended version of LBP 
The operations of LBP generate a histogram representation of picture which can be labelled as Cl (x, y) and can be explained as:
     Si = ∑x, yI (Cl (x, y) = i), i=0,1,…..,n-1
                Where, the different labels number is denoted by n.
I (A) = {_(0,A is false)^(1,A is true)
	Region Based Orb
The traditional ORB uses too much feature points in extractions to resolve the challenge what we are going to use is Region Based ORB. We update the ORB algorithm with base of dividing the region-based feature points in reference to the total point to be considered for feature extractions. The measure that we are going to take is as follow: 
	Starting with the division of the image into a square matrix that is X*Y of equal size. In this X and Y denotes the row and column respectively. The Scattered Points can be denoted as (S1, S2……………., SX*Y)
	The threshold is set as Th                 	                                                                           
                      Th = n/XY
Where, the total number of points considered is denoted by n.
	Recognition of features is done on every possible region and the considerations here are that in case we have a greater number of features than our threshold than we consider threshold as the number of feature else we have threshold lesser than we again degrade the threshold and call the process again and repeat it. 
	The non-maximal suppression approach is used to choose the best feature points, if we are having lesser value for n than the points considered as feature point.
	The number of points that are considered as the feature points is necessary to be lesser so that each of the region can be covered and feature can be classified out of them. 
	Analysis of CNN Model
In the approach that we are proposing consists of 4 different layers of convolution which has additional two layers of classifiers which are additional. In order to create a convoluted feature map, we have fused the features from vector along with the features from LBP. The layers of pools are using a static method for the conversion of activation in the phase when weighting of convolutional layer is there for the training purpose. We have used the rectified layer unit (ReLU) so that there is not any type of linearity in the whole network and which won’t affect the operational field of our respective convolutional layer. The efficiency of convolutional layers is determined as the loops. For efficiency add on we use back training in our model that is our output calculation is sent back to the training dataset which has no loss during the phase of back propagation. During the max pooling or convolution layer there are some chances of some layers to be lost in the scenario. The generated result is in spatial dimension which is nonlinear in nature and went through nonlinear down process. While during the pooling there is reduction in spatial size in which there is reduction of factor and computation that provides the required control over the fitting. At the last we again remap the two connected layers into a Pool feature map which again forms multi-dimensional structure after that it is converted into single dimension which is a vector and this vector is going to act as the feature vector in future operations. To continue the process of classification what we do is that we start treating the vector obtained as fully connected layer.
 
      Figure 2: Graphical representation of proposed method for FER.



	Rectification of Proposed CNN: In this phase we are not only considering rectification of our proposed method in spite we are also considering the rectification approach which automatically replaces and instead of the dataset used in the pre training approach it gets real time data and replace the previously existing layers with new layers that is generated after use of the model. We are doing this on the level of Kernel of pre training model which uses the technique of back propagation to send base layer back to the set of connected layers. The rectification parameters that are taken into consideration involves the resultant size of the convolutional layer are padding, stride, batch size, filter, sliding windows and the learning rate parameter. Padding in the case adds zeros to the bounds of the input. Stride controls over height parameters. In our approach our layers only have the task of training the model in the detailed feature classification which is based on block essence and in the end, we get the 7 emotions classification as a result of our model from our connected layers.                                                                                  
	Pipeline for proposed CNN.: Our pipeline architecture for the CNN has convolution along with pooling layers which are add on and in numbers we can have four layer of convolution and additional layers of two which are fully connected. The layer that we are considering as part of our sequenced structure contains the ReLU layer along with normalization that is batch based and a layer of dropout which completes our structure of fully connected layer. 
 
Figure 3: Architecture of Proposed CNN.


	Datasets
For the research, training and testing of our model we have gone through different datasets out of which some major contributors are FER2013 and JAFFE, beside this we have also checked for Cohn-Kanade extended version of dataset. We have used FER2013 mainly for the comparison of performance that how our model treat when there is not proper lighting condition while JAFFE is used to check for gender neutrality of our dataset. We have considered CK+ so that we can go for check of the images which are not initially in grayscale and as we came through testing. Beside this the major reason behind going for grayscale is grayscale dataset was in sufficient amount in compare to RGB but add-on feature of our model involves conversion of RGB into Grayscale as Grayscale images can be treated and operated much more efficiently rather than any other available dataset. Also, we have considered real time samples which involves video sample and some random images generated through the webcam. Initially we trained our model with the available dataset then we performed the testing for the real time and in return what we observed is our dataset is also changed which is due to our back propagation technique.
 
Figure 4: Seven basic emotion classification of FER 2013 Dataset. 
	How the facial emotion is recognized? 
The proposed approach has these basic steps:
	Preprocessing: In preprocessing initially the normalization of images is done which removes the differences and we get a good quality image to process further. After that our algorithm does gray scaling of the images and produces grayscale image to provide simplicity for the further work of model. Finally, in the phase of preprocessing we end up with re dimensioning in which we remove unnecessary portion of the image.

	Face Detection: In our proposed mechanism the second step is detection of faces. In face detection the cascade of images is used. In this we are using HAAR Cascade which is proved to provide high precision. HAAR cascade delete the non-usable objects that exists in the image and after that detects the ultimate face from it. We are considering rectangular characteristics at this phase.

	Feature Extraction 
In this approach we have used LBP as it is                      simple and highly powerful in classification. In this what we have done is:

	Division of images into smaller segments.
	Region based LBP histogram calculation and processing with the LBP code for each region separately.
	At the end the LBP feature obtained histogram is converted into single feature vector.
	Scheme of feature fusion
Along with fusion we have used the feature normalization technique to get better recognition. The normalization of LBP and ORB in our proposed methodology is been based on the formula:
Lb = l_b/(max⁡(l_b)),
Where lb denotes value of the features.
The LBP and ORB descriptors fusion is based on Z-Score methodology.
σ=∑_j^J▒〖(C_i-μ)〗^2 ,
μ=(∑_j^J▒f_j )/J,
〖~C〗_j= K x_(j-μ)/(σ+C)’
In this the Cj   denotes the LBP or ORB feature while ⁓Cj is the data of fusion feature used,  In this we have considered K as constant term whose value is 100 in this model.
	Emotion Classification:
Here our proposed model classifies the emotions into seven universal expressions from the dataset which are : Happy, Sad, Anger, Surprise, Disgust, Fear and Neutral. Before feeding the data to the CNN it is trained the step of emotion classification are as of the follow: 
	Data Splitting: In this dataset is split into training, public test and private test.
	Model training and generation: In this with per test there is new generation of dataset which are back propagated and fed again into CNN.
	Evaluation of Model: The dataset model as well as real-time images are used in the classification. 


	EXPERIMENTAL ANALYSIS AND RESULT

As we have used composite dataset for the feature performance, we obtained an accuracy above of 95 % and valid accuracy of 91%, which would be enhanced after several experiment approaches. To measure the performance of the system the results are measured using three metrics: accuracy rate, specificity rate and sensitivity rate. Truly-positive (TP), Truly-negative (TN), Falsely-positive (FP) and Falsely-negative (FN) are used to compute the characteristics.
Accuracy = (TN + TN)/(TP + FP + FN + TN),

Sensitivity = TP/(TP + FN )‘

Specificity = TN/(TN + FP)’

In our proposed model the total run time is of; 
O (n4 + n2) ↔ O (n4)   ⸪ Ɐn≥1 | n4 + n2 ≤ 2n4

Conclusion and Future Work 
We have used the indulging factors from LBP and ORB for betterment of CNN model. The dataset used were CK+, JAFFE and FER2013 all as grayscale images. In this we have found seven emotions and classified according to it.
In the analysis we have found it on average and has some look arounds to be improved in future with a greater number of experiments and datasets. The implementation majorly focused on the classifiers and the 4 layers which were working better with FER2013 dataset and provided quite adequate results in the whole.
Further more we are aiming to focus on mental issues of man or woman also with the emotion recognition mechanism and for that also we are training our model and indulging it to the practice.
 
Acknowledgment 
We would like to thank Department of Computer Science and Engineering, Jain Deemed to Be University’s staff for coordinating and helping us in the project completion. Specially we are thankful to our guide Professor Shashikala H.K who guided us and helped us in every possible way to accomplish this research. We would also be thankful to our classmates who helped us by motivating and being with us during the entire project. At the end we are thankful to our family who encouraged us and motivated us to do our level best in the project. This project is solely performed and sponsored by the department of Computer Science and Engineering, Jain Deemed to Be University.
The project was conceived in response to the privacy concerns raised by the sharing of patient medical information between hospitals and researchers. We try to show how the cutting-edge federated learning algorithm, which is extensively utilized in a number of commercial products that rely on user data, can help safeguard user privacy and can be used in the medical profession to train neural networks with data that researchers don't have access to. The concept we presented serves as a prototype for a broader system. This technology can be used and scaled to the point where it can revolutionize how medical researchers tackle challenges and work with other researchers from all around the world without violating data privacy rules. The classifier in this project is currently only capable of recognizing malaria parasites, but it may be trained to classify a variety of parasites for which tests are performed often.


	REFERENCES
 Zhang, M. Yi, J. Xu, R. Zhang and J. Shen, "Two-stage Recognition and Beyond for Compound Facial Emotion Recognition," 2020 15th IEEE International Conference on Automatic Face and Gesture Recognition (FG 2020), 2020, pp. 900-904, doi: 10.1109/FG47880.2020.00144.
L. Cai, J. Dong and M. Wei, "Multi-Modal Emotion Recognition From Speech and Facial Expression Based on Deep Learning," 2020 Chinese Automation Congress (CAC), 2020, pp. 5726-5729, doi: 10.1109/CAC51589.2020.9327178.
A. Poulose, J. H. Kim and D. S. Han, "Feature Vector Extraction Technique for Facial Emotion Recognition Using Facial Landmarks," 2021 International Conference on Information and Communication Technology Convergence (ICTC), 2021, pp. 1072-1076, doi: 10.1109/ICTC52510.2021.9620798.
C. Dalvi, M. Rathod, S. Patil, S. Gite and K. Kotecha, "A Survey of AI-Based Facial Emotion Recognition: Features, ML & DL Techniques, Age-Wise Datasets and Future Directions," in IEEE Access, vol. 9, pp. 165806-165840, 2021, doi: 10.1109/ACCESS.2021.3131733.
Ebenezer Owusu, Jacqueline Asor Kumi, Justice Kwame Appati, "On Facial Expression Recognition Benchmarks", Applied Computational Intelligence and Soft Computing, vol. 2021, Article ID 9917246, 20 pages, 2021. https://doi.org/10.1155/2021/9917246
Kumar Arora T, Kumar Chaubey P, Shree Raman M, Kumar B, Nagesh Y, Anjani PK, Ahmed HMS, Hashmi A, Balamuralitharan S, Debtera B. Optimal Facial Feature Based Emotional Recognition Using Deep Learning Algorithm. Comput Intell Neurosci. 2022 Sep 20;2022:8379202. doi: 10.1155/2022/8379202. PMID: 36177319; PMCID: PMC9514924.
Küntzler T, Höfling TTA, Alpers GW. Automatic Facial Expression Recognition in Standardized and Non-standardized Emotional Expressions. Front Psychol. 2021 May 5;12:627561. doi: 10.3389/fpsyg.2021.627561. PMID: 34025503; PMCID: PMC8131548.
Khan, Amjad Rehman. (2022). Facial Emotion Recognition Using Conventional Machine Learning and Deep Learning Methods: Current Achievements, Analysis and Remaining Challenges. Information. 13. 268. 10.3390/info13060268.
S. Giri et al., "Emotion Detection with Facial Feature Recognition Using CNN & OpenCV," 2022 2nd International Conference on Advance Computing and Innovative Technologies in Engineering (ICACITE), 2022, pp. 230-232, doi: 10.1109/ICACITE53722.2022.9823786.
L. Sun, J. Dai and X. Shen, "Facial emotion recognition based on LDA and Facial Landmark Detection," 2021 2nd International Conference on Artificial Intelligence and Education (ICAIE), 2021, pp. 64-67, doi: 10.1109/ICAIE53562.2021.00020.

 


