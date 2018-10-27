## Unsupervised Summary Generation of TED talk transcripts

#### Final project as part of the course 2018-CS/DS 823 - Introduction to Text Processing and Information Retrieval. 
#### We built an unsupervised summary generator and tested it on TED talk transcripts. The architecture consists of an autoencoder which is used to get a feature vector for every sentence, or sentence embedding. We then get distances between sentences in a transcript using these sentence embedding vectors, and generate a threshold value to decide which sentences to keep. 
#### The idea behind this approach is to to capture new 'concepts' that should be included in our summary.
#### Please read 'Project Report.pdf' for additional details. </br></br>

#### NOTE: You can get more details and code for training the autoencoder at https://github.com/Anshul-Gupta24/Neural-Machine-Translation.
# </br>

### Dataset
#### Download the file 'transcripts.csv' from https://www.kaggle.com/rounakbanik/ted-talks.
#### </br>

### Running the Code
#### To run the code:
#### >> python sent_dist.py </br>

#### Summaries are stored in 'summary\<ind\>.txt', where ind is the index of the transcript. To change the number of transcripts for which to generate a summary, modify the 'num_transcripts' variable in the main function. 
