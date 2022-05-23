# Net Purpose Machine Learning Coding Challenge

## Setup

- Ensure python is installed (I recommend anaconda for getting all the data science libraries pre-installed)
- Ensure docker is installed
- Download product\_sentiment.csv

## The task
This is a dataset from kaggle.com containing tweets about products and the sentiment of the tweet. 

Your task is to explore the data with some visualisations, and then train and "productionalise" a sentiment analyzer that you can input some text into and it will run your sentiment analysis model on it.
- This should be completable in 1 hour, and you're free to use any resources you find online. 
- Please cite (just a link or a site name is fine) references that you feel like you're using more than just a small part of.

### Subtasks
- Explore the data and come up with a visualization of which products have the most positive and negative tweets
- Train a classifier to determine the sentiment of the tweets (Positive/Negative/Neutral)
- "deploy" the classifier using a docker container and an endpoint of "/predict" to run predictions on new tweets
  - I'd recommend using flask as a simple server

## Submitting the task
When you have completed the challenge, please email me with a zip file or link to your solution to the email that is printed out of the following code:
```python
# Shamelessly stolen from https://www.stephendiehl.com/pages/hire.html
import os
import sys
import numpy
import string

if sys.version > '3':
    strl = str
else:
    strl = string

EMAIL = """
\xbb\xbb\xcc\xc9\xc8@\xc8\xbf\xce\xca\xcf\xcc\xca\xc9\xcd\xbf.\xbd\xc9\xc7
"""

y = string.ascii_letters
n = len(y)
M = numpy.identity(n, dtype=numpy.int32)
M[:n, n-1] = 1

a = numpy.array(list(map(ord, y)), dtype=numpy.int32)
x = ''.join(map(chr, numpy.dot(M, a)))
uncipher = strl.maketrans(x,y)

print( strl.translate(EMAIL, uncipher) )
```
