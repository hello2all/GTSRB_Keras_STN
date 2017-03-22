# Machine Learning Engineer Nanodegree
## Capstone Proposal
Liangyuan Tian  
Feburary 28th, 2017

## Proposal

### Domain Background
Recent advancement in autonomous vehicles has shown the once fictional technology is gradually becoming reality in the near future. In the foreseeable future, vehicles of various automation levels will coexist on the road. Thus it is crucial for automated driving machines able to recognise and interpret the meaning of traffic signs and follow traffic rules. Traffic sign recognition topics have been relatively well studied in research, notable publications include [[1]](http://orff.uc3m.es/bitstream/handle/10016/7089/traffic_escalera_IVC_2003_ps.pdf?sequence=1) which used nerual network to classify signs; and [[2]](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.63.283&rep=rep1&type=pdf) uses pattern matching; with the advances in convolutional nerual network, [[3]](http://s3.amazonaws.com/academia.edu.documents/31151276/sermanet-ijcnn-11.pdf?AWSAccessKeyId=AKIAIWOWYYGZ2Y53UL3A&Expires=1488356415&Signature=mNLKx2u9LSrucELj%2FDcqC598MVY%3D&response-content-disposition=inline%3B%20filename%3DTraffic_Sign_Recognition_with_Multi-Scal.pdf) is able to achieve human level accuracy using muti-scale convolutional nerual network.

### Problem Statement
For this multi-class classification problem, an automatic traffic sign recognition system will be built to categorize various traffic sign images. A deep convolutional neural network will be trained on various traffic sign images, its performance should exceed human accuracy to provide the highest safety for autonomous vehicles.

### Datasets and Inputs
[GTSRB [4]](http://benchmark.ini.rub.de/?section=gtsrb&subsection=news) is a multi-class, single-image dataset consists of 50,000+ traffic sign images with 40 classes collected in Germany. Some of the images smaples are shown below.

![sign sample](./footage/sample.png)

The system to be built in this project will use the images in the dataset as input, and the class category as output. Data can be downloaded from the official website [http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). These large amount of images are collect in real life, thus very suitable to be used as training data for the learning model.
### Solution Statement
The system will use the state of art convolutional nerual network with many advanced techniques include but not limited to image augmentation, image preprocessing, multi-scale convolutional nerual network, etc. Furthermore, other advanced methods such as spatial transformer network [[12]](http://papers.nips.cc/paper/5854-spatial-transformer-networks.pdf) will be experimented to achieve the best performance. The final accuracy of the model will be tested on the unseen test dataset percentage wise. 

### Benchmark Model
Many prior researches have been conducted on this dataset. The best published results are listed in the table below.

|Method|Accuracy|
|------|--------|
|Committee of CNNs[5]|99.46%|
|Color-blob-based COSFIRE filters for object recogn[6]|98.97%|
|Human Performance[7]|98.84%|
|Multi-Scale CNNs[3]|98.31%|
|Random Forests[8]|96.14%|
|LDA on HOG 2[9]|95.68%|
|LDA on HOG 1[10]|93.18%|
|LDA on HOG 3[11]|92.34%|

For the system to be built in this project, Human Performance[7] will be the selected benchmark model.
### Evaluation Metrics
The performance of the system is evaluated by prediction accuracy on test dataset, which is also provided on the official website. Addtionally, recall, precision, and F1 score will be utilized to measure the performance evolution of different archictecture and parameter sets.

### Project Design
The project will be organized as the following processes.

#### Data Exploration
The amount of data and its distribution will be analized to give insights how to handle the data.
#### Data Augmentation
Data augmentation is a standard practice on image data classification, this approach create more training data to help model improve. Common practices include sheer, scale, translate, brightness change.
#### Preprocessing
Experiment preprocessing methods such as normalization and black and white conversion.
#### Design Convolutional Nerual network
Various architecture of convolutional network will be experimented to gain the best performance. Techniques such as regularization, dropout, multi-scale features will be tested.
#### Training and Parameter Tuning
Various training methods and parameter sets will be tried out to optimize the model performance.
#### Experiments
Experiment more advanced techniques such as transfer learning or spatial transform network.
#### Summary
Write project report, summarize project achievements and further research directions.

### Reference
[1] De la Escalera, Arturo, J. Ma Armingol, and Mario Mata. "Traffic sign recognition and analysis for intelligent vehicles." Image and vision computing 21.3 (2003): 247-258.

[2] Miura, Jun, et al. "An active vision system for on-line traffic sign recognition." IEICE TRANSACTIONS on Information and Systems 85.11 (2002): 1784-1792.

[3] Sermanet, Pierre, and Yann LeCun. "Traffic sign recognition with multi-scale convolutional networks." Neural Networks (IJCNN), The 2011 International Joint Conference on. IEEE, 2011.

[4] J. Stallkamp, M. Schlipsing, J. Salmen, C. Igel, Man vs. computer: Benchmarking machine learning algorithms for traffic sign recognition, Neural Networks, Available online 20 February 2012, ISSN 0893-6080, 10.1016/j.neunet.2012.02.016. (http://www.sciencedirect.com/science/article/pii/S0893608012000457) Keywords: Traffic sign recognition; Machine learning; Convolutional neural networks; Benchmarking

[5] Multi-column deep neural network for traffic sign classification, Multi-column deep neural network for traffic sign classification, D. Ciresan, U. Meier, J. Masci, J. Schmidhuber, August 2012, Neural Networks (32), pp. 333-338

[6] Color-blob-based COSFIRE filters for Object Recognition, Color-blob-based COSFIRE filters for Object Recognition, Baris Gecer, George Azzopardi, Nicolai Petkov, 2017, Image and Vision Computing(57), pp. 165-174

[7] Man vs. computer: Benchmarking machine learning algorithms for traffic sign recognition, Man vs. computer: Benchmarking machine learning algorithms for traffic sign recognition, J. Stallkamp, M. Schlipsing, J. Salmen, C. Igel, August 2012, Neural Networks (32), pp. 323-332

[8] Traffic sign classification using K-d trees and Random Forests , Traffic sign classification using K-d trees and Random Forests , F. Zaklouta, B. Stanciulescu, O. Hamdoun, August 2011, International Joint Conference on Neural Networks (IJCNN) 2011

[9] Man vs. computer: Benchmarking machine learning algorithms for traffic sign recognition, Man vs. computer: Benchmarking machine learning algorithms for traffic sign recognition, J. Stallkamp, M. Schlipsing, J. Salmen, C. Igel, August 2012, Neural Networks (32), pp. 323-332

[10] Man vs. computer: Benchmarking machine learning algorithms for traffic sign recognition, Man vs. computer: Benchmarking machine learning algorithms for traffic sign recognition, J. Stallkamp, M. Schlipsing, J. Salmen, C. Igel, August 2012, Neural Networks (32), pp. 323-332

[11] Man vs. computer: Benchmarking machine learning algorithms for traffic sign recognition, Man vs. computer: Benchmarking machine learning algorithms for traffic sign recognition, J. Stallkamp, M. Schlipsing, J. Salmen, C. Igel, August 2012, Neural Networks (32), pp. 323-332

[12] Jaderberg, Max, Karen Simonyan, and Andrew Zisserman. "Spatial transformer networks." Advances in Neural Information Processing Systems. 2015.
