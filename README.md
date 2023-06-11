# Facebook Virality Prediction

This project is an attempt to predict "virality" of Facebook posts by an arbitrarily chosen Maltese news agency. The initial idea was to use the caption of a post to predict the engagement of a post, defined as

> engagement = reactions + shares + comments.

However, the problem was simplified to a Binary Classification model "Viral" and "Not Viral", which is simply a chosen threshold (in our case T=23), such that a post is viral iff `engagement > T`. 70% of TheMaltaIndependent posts have less engagement than 23. 

The model is far from perfect, with quite low precision/recall/F1-scores. However, some interesting insights are still observed. Two reasons why the model does not perform well are
- *Not enough data*. We are using only `n=2897` data points, and the number of features `m` is `m>>n`. This is because we are using NLP, and each unique word is essentially a dimension/feature.
- *Caption is not a strong predictor*: Perhaps other features like the sentiment of the post, the time of day it is posted etc are stronger predictors, which are not being taken into consideration. 
I believe that the first one is more likely, as `n=2897` is a far too small sample size for such a high-dimensional problem. It is challenging to gather much more data. 
## Data Sample
The data can be found [here](https://docs.google.com/spreadsheets/d/1mGNZX6qb7hMnKa9va_cyTjJm-B9RGV225U_JzEZryHw/edit#gid=0). A 3-row sample is

| Message                                                                                              | Created time             | Created date | Comments | Shares | Total Post Reactions | tot_engagement | engagement |
|------------------------------------------------------------------------------------------------------|--------------------------|--------------|----------|--------|----------------------|----------------|------------|
| An accident at the entrance to the Regional Road tunnels in St Julian's has left one person injured. | 2023-06-06T15:41:26 |   2023-06-06 |        0 |      4 |                   19 |             23 |          0 |
| A new Executive Chairperson of INDIS Malta has been appointed.                                       | 2023-06-06T12:56:49 |   2023-06-06 |        1 |      0 |                    0 |              1 |          0 |
| The latest Eurobarometer survey indicates that national discontent is growing.                       | 2023-06-06T11:32:17 |   2023-06-06 |        2 |      2 |                   100 |             104 |          1 |

## Methodology
The below steps were followed:
- **Data Cleansing**: POS tagging (take only nouns) stemming, lower case, replacing URLs with a placeholder word
- **Stop Words**: Remove common english words, along with those shown in only one caption, and words shown in more than 70% of posts
- **Use of up to 3-grams**, which manages to capture words/names which might have a significant impact such as "Joseph Muscat" or "Bernard Grech".
- **TfidVectorizer** to vectorize words in the corpus
-  **Oversample** due to class imbalance
- **Extra features** such day of the week (Monday, Tuesday etc) and the length of the post in number of words
- Fit **Logistic Regression** model
## Results
At the time being, only a linear model (Logistic Regression) using stochastic gradient descent was used as classifier. Considering the high-dimensional problem at hand, a linear classifier is the most prudent model to start with to avoid overfitting. Further investigation is required to confirm this. Other linear models with different loss functions were used with little to no success. 

Both $L^1$ and $L^2$ regularization were used, with the former showing slightly better results. Other parameters were left as the default values of the`sklearn.linear_model.SGDClassifier` class.

There is a class imbalance, as 30% are considered as viral using our definition. Oversampling showed a significant improvement to F1-score. The train-test split is chosen to be at a 70:30 ratio. The model runs almost instantaneously with around 3000 sample size at around 2 seconds. The numpy seed is set to 42 for reproducibility.

The below metrics are calculated based on the 30% testing data.

| Metric | Score  |
|--|--------|
| Accuracy | 62.76% |
| Precision | 40.00% |
| Recall | 45.45% |
| F1 Score | 42.55% |

- The accuracy is at a low value of almost 63%.
- The precision, recall and F1 scores are all at around a value of 42%. This is typically not a good sign, and suggests that the model needs serious improvement. This is challenging to do due to the limited test size provided. Below we can also see the ROC curve, and precision-recall curve. 
- Having said that, the values were compared to a DummyClassifier as a baseline model. Compared to this, the Logistic Regression model displays a significant improvement over this baseline model, indicating that there is some form of predictive power when it comes to the caption.
- Of the Non-Viral posts, 61% were correctly identified.
- Of the Viral posts in the test data, the model managed to guess 54% of them correctly. This is certainly not due to chance, as only around 30% of the test set contains posts with more than 23 engagement. 

![ROC Curve](https://github.com/DylanZammit/Facebook-Virality-Predictor/blob/master/img/ROC.png?raw=true)

## Analysis
As described above, the model can by no means be considered as "good", and various improvements can be applied. However, some interesting facts arise when looking at the features with the highest model coefficients. The model interprets posts with these keywords as more likely to get good engagement. Below we mention some of the most significant words and their corresponding posts 
### Hotel 
The most influential word was surprisingly non-political. However, taking a closer look at the Facebook posts containing this word, we can understand why such posts generate engagement. The majority of articles relate to the granting and withdrawal of licences for the construction of hotels locally. Unsurprisingly, this leads to a mixture of anger and delight from the public since this topic is quite contentious in the island due to our limited space. You can find the top post by engagement mentioning a hotel [here](https://www.facebook.com/597379732408318/posts/563235669156058).
### Bernice
A story that shook the country around the end of 2022 involves the murder of Bernice Cassar. Of the 14 posts containing her name, only 3 did not garner more engagement than 23. Unsurprisingly, the name of her husband, accused of the murder, also generated a high level of engagement. The top post by engagement can be found [here](https://www.facebook.com/597379732408318/posts/590912169721741).
### Politics
Other top-20 influential words include "Marie", "Repubblika President", "Party PN" and "Mark Camilleri". All of which are either political figures or entities. This is not surprising given how "passionate" or at least "opinionated" the Maltese population is when it comes to politics. One of the most engaged articles can be found [here](https://www.facebook.com/597379732408318/videos/573250945002185).

On the other hand, posts containing words such as "insurance" and "cannabis" tend to not perform as well as other topics.
## Improvements
Some points I can think of which would improve the model:
- Larger sample size
- Better data cleaning
- Testing out other ML models including random forests
- Include other features such as sentiment, topic (ex. politics, technology, entertainment, environment etc.), translation in case of non-English caption. Most of these would require including other ML algos in the pipeline
