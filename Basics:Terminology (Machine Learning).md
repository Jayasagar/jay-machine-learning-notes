# Basics/Terminology (Machine Learning)
#ML #AI #DL 
> TODO: Group concepts  

Terminology Words
* Sensor Fusion
* RFID readers

### What is ML
A computer program is said to learn from experience E () ‘with respect to some task T and some performance measure P’, if its performance on T, as measured by P, improves with the experience E.
* We can gain insight from a dataset;

::Machine learning is nothing but creating with linear algebra , and then improve them with calculus::


### Steps in developing a machine learning application
* Collect data.
* Prepare the input data.
* Analyze the input data.
* Train the algorithm: Build the Model
* Test the algorithm/Model

### Mapping function
function which can result output from given inputs.
Y = f(x) 
Y is ourput category and x is the one or more input attributes/features

### Supervised Learning
* In supervised machine learning an algorithm learns a model from training data.
* Prediction of correct values on applying the algorithms on given Dataset.
* Supervised learning problems can be further grouped into regression and ::::classification problems.
Classification problem 
Regression problem
Examples:
	* Problem1: Find out how many similar items we can sell over next 3 months from warehouse : Regression problem 
	* Problem 2: Find out software account is compromised or hacked? :  Classification problem

### UnSupervised Learning
No labels, Here is Dataset , plz find some pattern and group them(Cluster them)
	* Clustering Algorithm: 
		* News Google website:
		* Grouping customers by purchasing behavior.
	* Non-clustering:  
		* Cocktail party. Group by speakers  
	* Association: An association rule learning problem is where you want to discover rules that describe large portions of your data
		* People that buy A also tend to buy B
	* Density estimation algorithm
		* Do you need to have some numerical estimate of how strong the fit is into each group?
	* Examples
		* k-means for clustering problems.
		* Apriori algorithm for association rule learning problems
			* Before the event happens
			* We provide answer , based on idle scenario, like toss a coin: 50% head and tail 
		* Postperiori
			* After Actual experiment 
			* Real result after experiment
		* 

	* 

### Semi-Supervised Learning
* In reality , we do not think about Supervised and semi or unsupervised
* Start with Supervised and try to improve by moving towards semi-supervised
* Some data is labeled but most of it is unlabeled
* A good example is a photo archive where only some of the images are labeled, (e.g. dog, cat, person) and the majority are unlabeled.
*  Many real world machine learning problems fall into this area.
*  This is because it can be expensive or time consuming to label data as it may require access to domain experts. 
*  Whereas unlabeled data is cheap and easy to collect and store.

- - - -
> Both regression and classification problems belong to the supervised category of machine learning. In Supervised machine learning, a model or a function is learnt from the data to predict the future data.  
- - - -

### Classification 
* Things arranged by class or category.
A classification problem is when the output variable is a category, such as red or blue or disease and no disease.
* Is used to predict which class a data point is part of (discrete value). 
* Classification predicts the 'belonging' to the class.
Examples:
	## Name a unknown fruit 
	## Predict the weather : { Class->Sunny  DataSet->(Degrees:23, Wind: 70, Cloudy, etc…) }
	## Similarly the prediction of house price can be in words, viz., 'very costly', 'costly', 'affordable', 'cheap', and 'very cheap' : this relates to classification.

### Regression 
* A regression problem is when the output variable is a real value, such as dollars or weight.
* Is used to predict continuous values.
* where regression predicts a value from a continuous set,
Examples
	## How much my house should sold at price????
	## price of a house depending on the 'size' (sq. feet or whatever unit) and say 'location' of the house, can be some 'numerical value' (which can be continuous) : this relates to regression.

Start with Linear, 
### Linear Algorithms
Most of examples, match to Linear in life
We can 
* Logistic Regression: Giving probability (0 to 1)
* Linear Regression
* Linear Discrimanative Analysis

### Non Linear Algorithms
* K-Nearest
* Trees
* Naive Bayes
* SVM

### Parametric ML Algorithms
Usage the Assumptions to learn the model
* Parametric machine learning algorithms are often also called linear machine learning algorithms
*  Parametric or linear machine learning algorithms often have a high bias but a low variance
* An easy to understand functional form for the mapping function is a line, as is used in linear regression:
T0 + T1 × x1 + T2 × x2 = 0
* Where T0, T1 and T2 are the coefficients of the line that control the intercept and slope, and x1 and x2 are two input variables.
* Examples
	* Logistic Regression
	* Linear Discriminant Analysis
	* Perceptron
* Faster to train, require less data but may not be as powerful.

### Non-parametric ML Algorithms
No Assumptions to learn the model
* The method does not assume anything about the form of the mapping function other than patterns that are close are likely have a similar output variable.
*  Nonparametric or nonlinear machine learning algorithms often have a low bias but a high variance.
* Examples
	* Decision Trees like CART and C4.5
	* Naive Bayes
	* Support Vector Machines
	* Neural Networks
* Limitations
	* Require a lot more training data to estimate the mapping function
	* A lot slower to train as they often have far more parameters to train.
	* More of a risk to overfit the training data and it is harder to explain why specific predictions are made.
* 

### Learning Error/ Prediction error
can be broken down into bias or variance error.
There is no escaping the relationship between bias and variance in machine learning.   
*  Increasing the bias will decrease the variance.*  Increasing the variance will decrease the bias.

# Algorithm has Low Variance and High Bias
RoC curve 


### Bias Error
It Simplifies the assumptions made by the algorithm to make the problem easier to solve i.e. to make the target function easier to learn!
 Generally parametric algorithms have a high bias making them fast to learn and easier to understand.
*  In turn they are have lower predictive performance on complex problems that fail to meet the simplifying assumptions of the algorithms bias.
* Low-bias machine learning algorithms include: Decision Trees, k-Nearest Neighbors and Support Vector Machines.
* High-bias machine learning algorithms include: Linear Regression, Linear Discriminant Analysis and Logistic Regression.

### Variance Error
Sensitivity of a model to changes to the training data.
Variance is the amount that the estimate of the target function will change if different training data was used.

- - - -

The cause of poor performance in machine learning is either overfitting or underfitting the data.
- - - -

### Overfitting
namely how well the algorithm performs on unseen data.
Refer to [Introduction to Cross Validtions Types](Introduction to Cross Validtions Types)
[Machine Learning prediction accuracy statistics summary](https://jayasagar.github.io/machine%20learning/test%20data/accuracy/2017/12/25/Machine_Learning_prediction_accuracy_statistics.html)

### Underfitting

### Ensemble Algorithmsump

### Model/Algorithms
> Model = Algorithm(Dataset)  

### Predictive modeling
predictions made by the model

### Independent/Dependent variables

### Hypothetical Function
we have a way of measuring how well it fits into the data
h == function == which maps x’s value to y’s value

### Cost function 
	# h(x) = Theta 0 + Theta 1 (x)
	# Example = h(x) =. 1.5 + 0.5*(x)

### Error Rate



### Root Mean Squared Error
We can calculate an error score for our predictions called the Root Mean Squared Error or RMSE.
RMSE  = SQRT(SQ(pi - yi)/n)

p is the predicted value and y is the actual value, i is the index for a specific instance.

### Algorithm: Gradient Descent:
	# This is an algorithm for minimizing the cost function J
	# How it works
		## Start with some Theta values : J(0,0)
		## Keep changing the Theta values and and mimize the J value
		## Repeat until we find the minimum
	# Simultaneous Update of T0, T1 is important
	# We put θ0 on the x axis and θ1  on the y axis, with the cost function on the vertical z axis
	# Alpha == Learning Rate
	# Deviating term 

### Learning a best model
Different machine learning algorithms make different assumptions about the shape and structure of the function and how best to optimize a representation to approximate it. This is why it is so important to try a suite of different algorithms on a machine learning problem, because we cannot know before hand which approach will be best at estimating the structure of the underlying function we are trying to approximate.

### Nominal features
String values

### Continuous features
Numericvalues

#### Confusion Matrix
https://www-users.cs.umn.edu/~kumar001/dmbook/ch4.pdf
#### Hunt’s Arg
#### ID3, C4.5 and CART : Decision Tree arg
#### Linear Algorithms
Is based on method to achieve the best outcome (Maximum profit or lower cost)

### Cost function
Cost function is a way to define some weight/ priority score to a model developed.

#### Exponentiation
* Growth
* Raising a number to a power
* 2 power 3 === 8 === because 8 is the product of three factors of 2:

### Statistical Hypothesis
Example:
Claim: Uncertain Belief === Pizza delivery 30 mins or less
null hypothesis:  == Assumption is incorrect, says MAX delivery time 30 mins is incorrect. That is our assumption/ claim( Pizza delivery 30 mins or less) is not correct!

##### p-value
 helps to determine the significance of the results

You randomly sample some delivery times and run the data through the hypothesis test,


#### Entropy
* A numerical measure of the uncertainty of an outcome
* The higher the entropy, the harder it is to draw any conclusions from that information.
* Entropy quantifies the amount of uncertainty involved in the value of a random variable or the outcome of a random process.
* Is a measure of impurity
* Is broadly a measure of the disorder of some system

Mathematically, it can be calculated with the help of probability of the items as:
	H = - Sumof(p(x) logp(x))

#### Ensemble
A group of items viewed as a whole rather than individually:

#### Churn rate
The annual percentage rate at which customers stop subscribing to a service or employees leave a job.

#### Scatter matrix
#### Statistics
Merriam-Webster dictionary defines statistics as "a branch of mathematics dealing with the collection, analysis, interpretation, and presentation of masses of numerical data.

## heuristic
a commonsense rule intended to increase the probability of solving some problem. 
~Context:~ 
* Greedy algorithmic paradigm follows the problem solving heuristic.

## discrete
individually separate and distinct: speech sounds are produced as a continuous sound signal rather than discrete units.

## correlation
a mutual relationship or connection between two or more things: research showed a clear correlation between recession and levels of property crime.

## TIF/TIFF
A file with the TIF or TIFF file extension is a Tagged Image file

## What is parts of speech
In English the main parts of speech are noun, pronoun, adjective, determiner, verb, adverb, preposition, conjunction, and interjection.

## Corpus
"corpus" mainly appears in NLP area or application domain related to texts/documents, because of its meaning "a collection of written texts, esp. the entire works of a particular author or a body of writing on a particular subject."	
	## A corpus is a collection of machine-readable texts that have been produced in a natural communicative setting. 
	## Example: stopwords -> I, we, ours, my, you, etc…..
	## British National Corpus : 100 million words, Cross section of British English, spoken and written
## Corpora
is the plural for corpus.

## gazetteer
it's a list which would contain names of people, organizations, locations, etc. mined from various sources like Wikipedia and Freebase

## Morphology
A morpheme is the smallest unit of language that has meaning or function. prefixes, affixes, and other word structures that impart meaning.

## Priori
Right now
Example: A filter determines whether it is known a priori that the feature value is zero
## Concave
describes a surface that curves inward, or is thinner in the middle than on the edges.

## Convex
describes a surface that curves outward, or is thicker in the middle than on the edges.

## Inference
a conclusion reached on the basis of evidence and reasoning.

## Contour Lines
Draw the boundaries such that we can isolate the features or differentiations
![](Basics:Terminology%20(Machine%20Learning)/Screen%20Shot%202018-07-07%20at%205.17.30%20AM.png)


## Grayscale
 Images of this sort, also known as black-and-white or monochrome, are composed exclusively of shades of gray, varying from black at the weakest intensity to white at the strongest.

The input pixels are greyscale, with a value of 0.0 representing white, a value of 1.0 representing black, and in between values representing gradually darkening shades of grey.

## Pixel
A pixel is generally thought of as the smallest single component of a digital image.

Usually represented as a 

### Meaning of the word logits in TensorFlow
Reference to Google Developer Machine Learning.
[Machine Learning Glossary   |  Google Developers](https://developers.google.com/machine-learning/glossary/#logits)

In Math, **Logit** is a function that maps probabilities ([0, 1]) to R ((-inf, inf))

In ML, => The vector of raw (non-normalized) predictions that a classification model generates, which is ordinarily then passed to a normalization function. If the model is solving **a multi-class classification** problem, logits typically become an input to the softmax function. The softmax function then generates a vector of (normalized) probabilities with one value for each possible class.

### softmax
A function that provides probabilities for each possible class in a multi-class classification model. The probabilities add up to exactly 1.0. For example, softmax might determine that the probability of a particular image being a dog at 0.9, a cat at 0.08, and a horse at 0.02. (Also called full softmax.)

[Machine Learning Glossary   |  Google Developers](https://developers.google.com/machine-learning/glossary/#s)



#ml #ai #dl #basics #terminology 
