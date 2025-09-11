---
title: "My hands-on with NLP"
date: "2025-06-30"
theme: "Coding,ML"
summary: "Explanation of my pipeline including NLP, and what I've learned"
image: "https://raw.githubusercontent.com/Epwo/articles/refs/heads/main/images/nlp_exp/nlp_exp_header.jpg"
---

# The purpose
The purpose of this task, was to find and extracts categories from a csv containing feedback (verbtatims);
## example
to illustrate, if we take the disneyland review csv, we want to extract like; `parking problems` , `waiting line issues`, `safety` etc..


# The project ( topic-eur )
That is how the project was born, name is topic-eur because it is a topic crawler -> topic-eur (which also is a pokemon!)

I did not really have a plan or hints that were given to me, so it was a fun research data science project.

# The idea
My idea was as follow:
To get the meaning of each verbatim, I choose to use the transformers-based models.
The main supposition I made was that, if we look at the highest values attentions score of the sentence we could extract its meaning.
This is an idea based of the human way of learning a new language, if I try to learn german. I am really bad at german, therefore I can't understand nor translate word by word, I will however be able to understand the overall meaning of the sentence. That is because I can pick up 2 or 3 words around the subject and an adjective.
![alt text](https://raw.githubusercontent.com/Epwo/articles/refs/heads/main/images/nlp_exp/image.png)

Therefore, if we look at the attention layer of a transformer-based model this should give us an idea of the words contaning most of the meaning of each verbatim.
# The model 
Because our verbatims will be in french, I choose to go with the latest camem-BERT model ( made by INRIA )
[moderncamembert](https://huggingface.co/almanach/moderncamembert-base)

So we will for each row of the given csv ( each row representing a verbatim )
![step one of the process](https://raw.githubusercontent.com/Epwo/articles/refs/heads/main/images/nlp_exp/step1.png)

# the Attention Layer
However, if you know the transformers-based models quite well. You know that there are two mains issues with just going with the data as it is.
## 1st issue
 the first issue is that thoses models are trained to work on long sentences and even large texts. So they dont stop at just the "." at the end.
 To do that, they put a lot of attention on specials tokens that the tokenizer gives them to notify that this is the end of sentence for example.
 SO we will remove thoses. we alos will remove the words that could be meaningful but in a group, not alone. like the stopwords or the punctuations.
## 2nd Issue
We also have another problem with the tokenizer, his main purpose is to divide the text into 'tokens' that are words or part of a word. This last part, is what we want to get rid of. Let's take the word "magnifique" as an example > if "magnifique" becomes "magn-", "ifi-","que" . The token that contains the most meaning might be only one of them. However it is the word that contains the most meaning not only the "ifi-" ( which is btw, non usable as it is).
Luckly for us, the tokenizer when splitting words doesnt just cut them in half, he also add a little "thing" to say that this word was cut by the tokenizer.
Which means that we can recreate thoses words !
(For camembert it is actuallly splitting the words like this : lamentable ->  lam, ##entab, ##le)
So we will recreate the words that have been split, and add their attention score.

![the steps to "purify" the data](https://github.com/Epwo/articles/blob/main/images/nlp_exp/purify_data.png?raw=true)

# Data science
Now we have as the output of the first step, a dict of each verbatim, where each one has the 5 most "meaningful" words ( the tensor value & the word itself )
The idea is that, the model (here the BERT-based model) is determinist. He will (roughly) give the same emplacement in the latent space for two words that have the same meaning.

However, BERT is giving a 758 dimensions tensor as an input. The issue is that we as humans, cant represent this many dimensions. And that the usal data sciences algortihms are not fit for this many dimensions. ( or at least what I will be using.)
this is where umaps comes into play
## UMAP
UMAP is pretty robsut algorithm used to reduce the dimensions :
![umap_image](https://miro.medium.com/v2/resize:fit:1400/1*fGQImmija7kepddB7SFaGA.jpeg)

it is known to be quite good at loosing less informations in the process than the other methods ( at least from what I've read. )
I choose to go with UMAP, for the dimensions reduction. We are going from 759 to 3, to be able to make a 3D graph representation of the words.

## Outliers
As often when dealing with big data, it appears to have what we call outliers.
Thoses are values that are not pertinent, and can often disturb the clustering algorithms. 
![example of an outlier](https://github.com/Epwo/articles/blob/main/images/nlp_exp/outliers.png?raw=true)

I have choose to go with KNN to remove the outliers.

## The (real) clustering
Durings my research to find the best algorithm to clusterise my type of data. I've stumble upon DBSCAN, which ,I think is, quite awesome
![dbscan](https://github.com/Epwo/articles/blob/main/images/nlp_exp/dbscan_general.png?raw=true)

DBSCAN, base itself on two values defined by the user, the 'eps' value and the 'min_samples' value.
If we take a first point A, the 'eps' will be the size of the range to allow another point B to be classifed in the same cluster (A+B) (see the schematic I drew below)
The 'min_values' is simply the number of points a group needs to have (minimum) to be consider a real cluster.
![how it works](https://github.com/Epwo/articles/blob/main/images/nlp_exp/dbscan_detailled.png?raw=true)

The awesome thing with this method, is that the user virtually dosent have to choose the number of clusters. (because in my case, this number is quite changing, depending on the csv we are analyzing)
However, while we don't have to choose a number of clusters. We still need to define the value of both 'eps' and 'min_values'. 

### Choosing the 'range'(eps) and 'min_values'
I've tried to choose a fixed values for both. But while testing with other datasets (other sets of verbatims), I saw that it was differing too much, from one dataset to another.

Then I thought, that thoses values would be a simple linear regression. (like this :)
![alt text](https://github.com/Epwo/articles/blob/main/images/nlp_exp/reglin.png?raw=true)

Turns out, it was not a linear regression either.
So i thought, what could i have and what could i use ?
I want to guess the 'range' (eps) and the 'min_values'. E,N
What values ( that changes, with each dataset) am i left with ? : the sigma value (standard deviation) between the points, and the total number of points (the numbers of words)

Therefore I was left with something like this : 
f(nb_pts,sigma) = eps 
g(nb_pts,sigma) = min_values

So to compute this f and g function, I choose to train a Random Forest model ( a Machine learning solution, amongst many)
The hard part was, on what to train it. I choose to fine-tune at hand, the eps and min_values for about ten datasets 
(I know that this is not significant, but it was slow and painful. and I tried to make them as representative as possible, like at each extremums)

NB : Now that I am writing this, I could have gone with a Reinforcement learning, where you dont need to input any "train data".

Once the two models were trained, I could print out some pretty graphs that looks like this
![alt text](https://github.com/Epwo/articles/blob/main/images/nlp_exp/RF.png?raw=true)


# A representation, of the data
At this point the hardest is behind us, it will be more straight forwards starting from here.
But first, to vizualise better what I was doing up to this point, and to see if the clustering method with my predicted eps ans min_values are looking good.
![alt text](https://github.com/Epwo/articles/blob/main/images/nlp_exp/visualize.png?raw=true)

> [!Info]
> Because there are too many clusters compared to the number of colors availbles in plotly, some clusters might look like they are from the same cluster, despite beeing in 2 different ones, but the color is the same, because the color is chosen randomly among a "short" list.

*Work in progress*