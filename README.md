# RadioML_complex
A complex-valued neural network (CVNN) framework is currently under development to advance research on neural networks taking a complex-valued input
(most research in neural networks uses a real-valued NN) without changing the complex value into a vector of two real values, 
and thus processing real values into the NN. 
This preliminary work outputs a baseline program to test complex-valued datasets performance. 

RadioML datasets
Complex-valued signals are largely used in wireless communications. Thus, developing neural networks as a general approximation function of a filter or 
a signal processor can be useful and/or necessary. The RadioML dataset is a 2-channel real-valued data generated to represent radio frequency modulations.
Each input data point represents a complex number in a vector form where one value is the real part and the other is the imaginary part. 
The dataset are generated using wireless communications simulation and/or test environment. Several versions of the dataset are available online

The Python files classify radio signal from the RadioML dataset hosted at https://www.deepsig.ai/datasets under the name RML2016.10a.tar.bz2.
The .png files are classification results after dividing the dataset into the three raining, Validation and Test subsets.

References

[1] J. A. Barrachina, C. Ren, C. Morisseau, G. Vieillard, J. P. Ovarlez, “Complex-Valued vs. Real-Valued Neural Networks for Classification Perspectives: 
An Example on Non-Circular Data,” 2019, https://arxiv.org/abs/2009.08340

[2] R. Chakraborty, Y. Xing and S. X. Yu, " SurReal: Fréchet Mean and Distance Transform for Complex-Valued Deep Learning,"2019,  
https://arxiv.org/abs/1906.10048,
