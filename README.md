# Facebook Virality Prediction

This project is an attempt to predict "virality" of Facebook posts by an arbitrarily chosen Maltese news agency. The initial idea was to use the caption of a post to predict the engagement of a post, defined as

> engagement = reactions + shares + comments.

However, the problem was simplified to a Binary Classification model "Viral" and "Not Viral", which is simply a chosen threshold (in our case T=80), such that a post is viral iff `engagement > T`.

The reason for failure could be one of two:
- *Not enough data*. We are using only `n=987` data points, and the number of features `m` is `m>>n`. This is because we are using NLP, and each unique word is a dimension/feature.
- *Caption is not a strong predictor*: Perhaps other features like the sentiment of the post, the time of day it is posted etc are stronger predictors, which are not being taken into consideration. 
I believe that the first one is more likely, as `n=987` is a far too small sample size, but it is challenging to gather much more data. 
## Data Sample
The data can be found [here](https://docs.google.com/spreadsheets/d/1mGNZX6qb7hMnKa9va_cyTjJm-B9RGV225U_JzEZryHw/edit#gid=0). A 3-row sample is

| Message                                                                                              | Created time             | Created date | Comments | Shares | Total Post Reactions | tot_engagement | engagement |
|------------------------------------------------------------------------------------------------------|--------------------------|--------------|----------|--------|----------------------|----------------|------------|
| An accident at the entrance to the Regional Road tunnels in St Julian's has left one person injured. | 2023-06-06T15:41:26 |   2023-06-06 |        0 |      4 |                   19 |             23 |          0 |
| A new Executive Chairperson of INDIS Malta has been appointed.                                       | 2023-06-06T12:56:49 |   2023-06-06 |        1 |      0 |                    0 |              1 |          0 |
| The latest Eurobarometer survey indicates that national discontent is growing.                       | 2023-06-06T11:32:17 |   2023-06-06 |        2 |      2 |                   100 |             104 |          1 |

## Methodology
The below steps were followed:
- **Data Cleansing**: stemming, lemmatization, lower case, replacing URLs with a placeholder word
- **Stop Words**: Remove common english words, along with those shown in only one caption, and words shown in more than 30% of posts
- **Use of 2-grams**, which manages to capture words/names which might have a significant impact such as "Joseph Muscat" or "Bernard Grech".
- **TfidVectorizer**
-  **Oversample** due to class imbalance
- **Extra features** such day of the week (Monday, Tuesday etc) and the length of the post in number of words
- Fit **Logistic Regression** model
## Results
At the time being, only a linear model (Logistic Regression) using stochastic gradient descent was used as classifier. Considering the high-dimensional problem at hand, a linear classifier is the most prudent model to start with to avoid overfitting. Further investigation is required to confirm this. Other linear models with different loss functions were used with little to no success. 

Both $L^1$ and $L^2$ regularization were used, with the former showing slightly better results. Other parameters were left as the default values of the`sklearn.linear_model.SGDClassifier` class.

There is a class imbalance, as only 8.5% are considered as viral using our definition. Oversampling showed slight improvement to accuracy, so is  implemented. The train-test split is chosen to be 75%. The model runs almost instantaneously with around 1300 sample size (due to oversampling). The numpy seed is set to 42 for reproducibility.

The below metrics are calculated based on the 25% testing data.

| Metric | Score |
|--|--|
| Accuracy |  87.8%|
| Precision |  0.3913|
| Recall |  36%|
| F1 Score |  0.375|

## Analysis
- The accuracy is at a good value of almost 88%, however the recall is at 36%, which indicates that there is *some* predictive power in the model, although not great either. 
- The low-values of precision and F1-scores are not as promising, both being less than 0.4. Below we see the ROC curve, and precision-recall curve. 
- Of the Non Viral posts, 94% were correctly identified. However, this is not as impressive due to the class imbalance
- Of the Viral posts in the test data, the model managed to guess 36% of them correctly based on the caption and the day of the week. This might be considered as "good" depending on the context and user. Correctly identifying a third of your posts as "viral-worthy" might be considered useful 

![ROC Curve](https://github.com/DylanZammit/Facebook-Virality-Predictor/blob/master/img/ROC.png?raw=true)

## Improvements
Some points I can think of which would improve the model:
- Larger sample size
- Better data cleaning
- Testing out other ML models including random forests
- Include other features such as sentiment, topic (ex. politics, technology, entertainment, environment etc), translation in case of non-English caption. Most of these would require including other ML algos in the pipeline
