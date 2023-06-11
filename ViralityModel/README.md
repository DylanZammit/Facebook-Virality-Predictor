# Running the script
Firstly, download the necessary python packages using
> pip install -r requirements.txt
> 


To reproduce the results in the readme of the root page, the following command needs to be run.

>python model.py --oversample --classifier SGD --threshold 0.5 --viral_threshold 23

You must also download the necessary `nltk` files by uncommenting on the first run
```
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')
```