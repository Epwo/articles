---
title: "My hands-on with NLP"
date: "2025-06-30"
theme: "Coding","ML"
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
