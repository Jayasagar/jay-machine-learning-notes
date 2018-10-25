# Math basics (Machine Learning)
#maths #calculus #ml #AI #ML #mathematics

- - - -

Linear algebra == 35%
Probability Theory and Statistics == 25% 
Multivariant Calculus == 15% 
Algorithms and concepts == 15% 
Other = 10%

::Machine learning is nothing but **creating** with linear algebra , and then **improve and optimize** them with calculus::

>   
###### Deep into Linear Algebra
> The Matrix Cookbook (Petersen and Pedersen, 2006).  
> https://cosmathclub.files.wordpress.com/2014/10/georgi-shilov-linear-algebra4.pdf  

- - - -
# Maths in Real World
### Pure Mathematics
	* It is scientifical work, where people try to find new formulae, new theory and try to research and explore Mathematical world. 
	* They don’t focus on Real world use cases
	* Example: MandelBrot equation invention

###### Number System
	* We can do sum, product, divide operations on the numbers
	* Prime numbers, PI, Exponential number
	* Real numbers, Rational numbers, Complex numbers
	* Quaternion, Octonion and Cardinal numbers
	* Infinity

###### Structures
	* This is where we start putting them into EQUATIONS using Variables
	* y = mx + b

###### Algebra
	* Contains the rules how we **manipulate** these equations (y = mx + b) 
	* We also use Vectors, Matrices and Tensors as a multi dimensional Structures 
###### Linear Algebra
	* Rules how Vector, Matrices and Tensors relate to each other is Linear Algebra.

###### Number Theory
	* Properties of Prime Numbers

###### Combinatorics
	* Looks at the certain structures like Trees, Graphs 

Partition Theory, Order Theory(Factors of 30) and Group Theory(Cubic)

###### Spaces
	* Geometry, Pythagoras
	* Fractal Geometry: Scale in variant:  e.g. MandelBrot set
	* Topology: Mobius Strip 
	* Measure Theory
	* Differential Geometry

###### Changes
Calculus
	* Integrals, Differentials , Derivatives
	* Gradient of functions 
Vector Calculus
	* Dynamic Systems
		* Fluid Flows
			* Eco system:  Feedback loops
	* Choas Theory
		* Butterfly Effect
	* Complex Analysis

### Applied Mathematics 
	* Spend time to identify the use case in the field of science, Physics and find the real world solutions to the problems.
	* Solve real world problems is the goal

###### Areas
	* Engineering 
		* Control Theory
	* Physics
	* Game Theory
	* Numeric Analysis
	* Biology
	* Probability
	* Statistics
	* Mathematical Finance
	* Economics
	* Chemistry
	* Optimizations
	* Computer Science
	* Machine Learnings
		* Linear Algebra, Dynamical system, Optimizations 
	* Crytpography

![](Math%20basics%20(Machine%20Learning)/Screen%20Shot%202018-07-09%20at%206.41.08%20AM.png)


![](Math%20basics%20(Machine%20Learning)/Screen%20Shot%202018-07-09%20at%206.57.16%20AM.png)

- - - -


# Explain WHY and How maths being used through House price prediction example!
* Statistics at core, used for analytics
	* Extracts the useful information from Data
	* Relation between variables
	* Statistical inference technique: Linear Regression
		* 
* Calculus is for improve and optimize the models
	* Example: Calculate error rate in Linear Regression
	* Y = mx + b , in this equation , calculus helps to find the m and b variables such that we have least error possible to predict the out put Y
	* 
* Linear Algebra is for running models feasible on massive datasets
	* Multivariant Regression
	* it is concerned with working with multiple input features
	* Linear algebra helps
	* 
* Probability is for Likely wood of an possible outcome
- - - -


### Understanding Vectors

![](Math%20basics%20(Machine%20Learning)/Screen%20Shot%202017-12-31%20at%205.11.02%20PM.png)

In above picture, A is Vector, A with pipe symbols is A magnitude
A = [3,4] , |A| = 5

### Gaussian distribution(Normal distribution)

![](Math%20basics%20(Machine%20Learning)/GaussianDistribution.gif)
Normal distributions can differ in their means and in their standard deviations. Figure 1 shows three normal distributions. The green (left-most) distribution has a mean of -3 and a standard deviation of 0.5, the distribution in red (the middle distribution) has a mean of 0 and a standard deviation of 1, and the distribution in black (right-most) has a mean of 2 and a standard deviation of 3. These as well as all other normal distributions are symmetric with relatively more values at the center of the distribution and relatively few in the tails.


**Seven features** of normal distributions are listed below. These features are illustrated in more detail in the remaining sections of this chapter.

* Normal distributions are symmetric around their mean.
* The mean, median, and mode of a normal distribution are equal.
* The area under the normal curve is equal to 1.0.
* Normal distributions are denser in the center and less dense in the tails.
* Normal distributions are defined by two parameters, the mean (μ) and the standard deviation (σ).
* 68% of the area of a normal distribution is within one standard deviation of the mean.
* Approximately 95% of the area of a normal distribution is within two standard deviations of the mean.

[Introduction to Normal Distributions](http://onlinestatbook.com/2/normal_distribution/intro.html)
- - - -

### Root Mean Squared Error
We can calculate an error score for our predictions called the Root Mean Squared Error or RMSE.
RMSE  = SQRT(SQ(pi - yi)/n)

p is the predicted value and y is the actual value, i is the index for a specific instance.

### Conditional probability

Total = 7 Stones
Black = 4 
Gray = 3

P(B) = 4/7 and P(G) = 3/7

![](Math%20basics%20(Machine%20Learning)/Bayes_rule.png)

P(c|x) is the posterior probability of class (target) given predictor (attribute). 
P(c) is the prior probability of class. 
P(x|c) is the likelihood which is the probability of predictor given class. 
P(x) is the prior probability of predictor.

Where x is the input attribute and c is the class category

### Quartile (First and Third)
The lower quartile value is the median of the lower half of the data. The upper quartile value is the median of the upper half of the data.

Refer here for clear explanation: [First Quartile and Third Quartile](http://web.mnstate.edu/peil/MDEV102/U4/S36/S363.html)
##  Interquartile range or IQR
![](Math%20basics%20(Machine%20Learning)/simple.box.plot.defs.gif)

## Mean
Mean and Average are synonyms !!!
## Median
The median is the middle point of a number set, in which half the numbers are above the median and half are below. In our set above, the median is 30. But what if your number set has an even number of, er, numbers:

11
23
30
47
52
56
To calculate the median here, add the two middle numbers (30 + 47) and divide by 2. The median for our new list is 38.5.

## Standard Deviation
Source : [Standard Deviation and Variance](http://www.mathsisfun.com/data/standard-deviation.html)

Example: Height of the Dogs

The heights (at the shoulders) are: 600mm, 470mm, 170mm, 430mm and 300mm

Mean = Avg of 5 values = 

Variance = Take each difference, square it, and then average the result:
i.e. => Square(600 - Mean) + …. / 5 

> Standard Deviation is just the square root of Variance.  
> Sqrt(Variance)  

![](Math%20basics%20(Machine%20Learning)/statistics-standard-deviation.gif)

147 is the Standard deviation!!!!

So, using the Standard Deviation we have a "standard" way of knowing what is normal, and what is extra large or extra small.

## Slope or Gradient
Is a NUMBER that describes the direction and the steepness os the LINE.
“Rise over Run”

![](Math%20basics%20(Machine%20Learning)/Screen%20Shot%202017-11-11%20at%207.29.51%20AM.png)

Credit: [Slope - Wikipedia](https://en.wikipedia.org/wiki/Slope)
Δ, is commonly used in mathematics to mean "difference" or "change".

## Y-intercept
![](Math%20basics%20(Machine%20Learning)/Screen%20Shot%202018-03-09%20at%207.01.09%20AM.png)
When you have a linear equation, the y-intercept is the point where the graph of the line crosses the y-axis. 

## Differentiation
Differentiation is the action of computing a derivative.

## Derivative
Reference: [What is Calculus?  (Mathematics) - YouTube](https://www.youtube.com/watch?v=w3GV9pumczQ)
[Derivative - Wikipedia](https://en.wikipedia.org/wiki/Derivative)

Measures the sensitivity to change of the output value with respect to the change in its input value/argument.

Derivative is often described as the "instantaneous rate of change" 


## Linear
Involving one dimensional only
Able to represented by a straight line on a graph.
Progressing sequentially in a line;  
E.g.  Linear search: search for an element in a sequence, one by one element until element found.

## Linear function 
f = mx + k
f is the a linear function of x, then coefficient of x is m(slope)  


## Proportion
A  part, share, or number considered in comparative relation to a whole
::Example::: 
Total Apples = 10, 
Class_label_category: Bad= 3 and Good = 7
Proportion = count(class)/count(total)
Bad Apple Proportion = 3/10 = 30% share
Good Apple Proportion = 7/10 = 70% share

## Exponentiation/Power
![](Math%20basics%20(Machine%20Learning)/Screen%20Shot%202018-08-22%20at%205.35.53%20AM.png)

Power: Where base is variable
Exponential: Base is constant
Growth
Raising a number to a power
2 power 3 === 8 === because 8 is the product of three factors of 2:
2 power of -1 == 1/2
In the equation y = logb x, the value y is the answer to the question "To what power must b be raised, in order to yield x?". 
From above b(power of) y === x

## Logarithm
Is the inverse operation to exponentiation, just as division is the inverse of multiplication and vice versa.  
Logarithms are valuable for describing algorithms that divide a problem into smaller ones
Real application / example
	## Human Salary 10(power)5
	## Distance Earth to moon 10(power)8
	## Number of cells in human body 10(power)14
Howdy you represent huge numbers on GRAPH ??? That is where logs come into rescue
If events vary drastically , then we could use logs
	## Richter scale :: Each quake 
	## Is a common log(Base 10) represents to understand the earth quake a place 
	## 

# Linear Algebra
Its all about 
	* operations on collection fo data , such as Array
	* Relationships are like Algebra, look at X and wonder what is Y
	* Scalar, Vector, Matrix and Tensor
	* Sum, Product, Transpose, Inverse, Identity operations 

Understand Matrix, Inverse Matrix, Identify Matrix 

###### Linear Dependence and Span

Linear dependence means, any vector in the given set has a co relation with the other vector.

Example
![](Math%20basics%20(Machine%20Learning)/Screen%20Shot%202018-07-11%20at%205.06.01%20AM.png)
![](Math%20basics%20(Machine%20Learning)/Screen%20Shot%202018-07-11%20at%205.30.13%20AM.png)


::IF and ONLY IF all arbitrary ‘C’ are proved to Zero === Linear independent::

###### Linear Independent 
![](Math%20basics%20(Machine%20Learning)/Screen%20Shot%202018-07-11%20at%205.23.27%20AM.png)

In the above attachment, example Linear independence example, where all arbitrary ‘C’ are proved to Zero.

::IF and ONLY IF all arbitrary ‘C’ are proved to Zero === Linear independent::

###### Diagonal matrices

Formally, a matrixDis diagonal if and only if Di,j= 0 for all I != j

###### Symmetric Matrix

###### Unit Vector

###### Orthogonal

