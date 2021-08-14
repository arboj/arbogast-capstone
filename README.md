# Mapping Natural Disaster Locations from Social Media Content
 
### Overview

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; This project explored a workflow and method to condition, classify, geolocate and map unstructured data from social media in order to discover clusters of locations mentioned in natural disaster social media posts. This method collected posts from Twitter leveraging the **[Snscrape](https://github.com/JustAnotherArchivist/snscrape)** python library, and processed and conditioned the text of the posts. 

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; A binary classification recurrent neural network was trained in TensorFlow with Keras using human labeled tweets aggregated by the **[CrisisBenchmark dataset](https://crisisnlp.qcri.org/crisis_datasets_benchmarks)** created by the **[Crisis NLP project at the Qatar Computing Research Institute](https://crisisnlp.qcri.org)**. 

The text of the training data was vectorized using a pre-trained text embedding built from tweets using the Global Vectors for Word Representation (GloVE)  methodology and available from the **[GloVe Project at Stanford University](https://nlp.stanford.edu/projects/glove/)** (Pennington, Socher, and Manning 2014). Hyperparameters for the model were tuned with the Hyperband algorithm, and the final model was evaluated using a 5-fold cross-validation. The model and text embedding built from the training data were used to classify tweets pulled from June and July 2021 for natural disaster informativeness. 

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; The resulting informative tweets were then parsed for location and georeferenced using the **[Mordecai python library](https://github.com/openeventdata/mordecai)**. Finally, the informative posts with locations mentioned in the text were displayed in an interactive R shiny web application for end users to map location explore by  filtering data by geography, date and time, and research interest; discover trending topics on their selected data in a word cloud and export data for use by analysts in further research and visualization.



### Scrape, Clean, Predict and Geoparse
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Clone this repository to your local machine, there are several very large files such as the model as well as the tweets scraped and geo refereneced from June and July 2021, so pulling the first clone will take several minutes.

The 




### Helpful other links
Docker									https://www.docker.com
Docker was used it initiate a elastic search over a geonames index for the mordecai geo parser
Elastic 		https://www.elastic.co/guide/en/elasticsearch/reference/current/docker.html
Elastic was used to enable Mordecai to search the geonames index in a docker container




### Jupiter Notebooks Used for testing and data wrangling

Exploration of Variables			[TextDataEDA.ipynb](https://github.com/arboj/arbogast-capstone/blob/c9bfe55972b65e40304e620bea2b03d45ec51169/Code/TextDataEDA.ipynb)

Text Preprocessing			[TextConditioningandMachineLearning.ipynb](https://github.com/arboj/arbogast-capstone/blob/f81faeac7c3be1436587886de6bee341629b2453/Code/TextConditioningandMachineLearning.ipynb)













