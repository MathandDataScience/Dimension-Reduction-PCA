# Dimension-Reduction-PCA
## Principal Component Analysis of the Bubble Nebula, A in-depth look at the HST project 07515 01 WFPC2 files

### Introdution
Often data sets have many dimensions or contain many variables with many samples.
As data scientists we are asked to make sense of this data and find something useful from it.
This can be a hard problem to approach without trying to see how these samples or dimen-
sions relate. As human we have a hard time understanding something beyond 4 dimensions
as this is what we experience in our life span. When it comes to visualizing the data, we often
won’t see a pattern in more than two dimensions and maybe three dimensions depending
on the data. There are a few ways to approach the visualization of a data set for example
scatter plots and heatmaps. Scatter plot are a way of visualizing the relationship between
two variables. Each data point is plotted as a point on the x-y plane, with one dimension
represented on the x-axis and the other dimension represented on the y-axis. This is repeated
for all dimensions of the dataset. These scatter plots can be useful for identifying patterns
or trends in the data, such as linear or nonlinear relationships between dimension (or vari-
ables). Heatmaps, on the other hand, are a way of visualizing patterns in the data through
color-coding. Heatmaps can be used to visualize the same relationships as the scatter plot.
Understanding if the relationship is linear or nonlinear is not possible using heatmaps. This
is why heatmaps are often used in conjunction with clustering algorithms to identify groups
of similar items or features. when working with large dimensional data these scatter plots
and heatmaps become hard to read or understand the treads of the data. Along with this
fact, the key factor that these visualization techniques leave out is how much each dimension
influences the outcome of the data. This can be understood through Principal Component
Analysis or PCA and a scree plot. PCA works by identifying the directions in the “vari-
able space” that account for the most variance in the date (i.e. principal components).[1]
Understanding this will allow the user to determine what dimensions matter the most and
reduce or remove the dimensions that don’t contribute to the data substantially. Using the
principal components that contribute the most, allows for the projected of the data onto
a lower-dimensional space maintaining the majority of the data’s features. The data more
easily be plotted to understand the data further. The inter-working and mathematics of
this will be discussed later. The problem at hand utilizes PCA in a practical way to reduce
six grayscale images into one color image. The data utilized come from the Hubble Legacy
Archive(HLA) this database holds the collection of the raw fits data from the Hubble Space Telescope (HST). [4]

### Background
 A few important mathematical techniques need to be discussed in depth, this includes PCA, eigen decomposition, Gram-Schmidt and QR factorization. Along with these mathematical techniques the nature of the sensor and human eye needs to be discussed. As mentioned in the introduction principal component analysis (PCA) serves as a method for understanding trends in data and can be used to represent the data in an orthogonal basis. It is also an important technique used for dimension reduction.[2] Conventionally PCA is preformed but finding the covariance matrix of the data $\Sigma \in \mathbb{R}^{MxM}$ this is found by multiplying the data $A\in \mathbb{R}^{MxN}$ with the data transported $A^T\in \mathbb{R}^{NxM}$.
 
$$\Sigma = AA^T$$

This outer product forms the covariance matrix 

$$ \Sigma  = \begin{bmatrix}
var(x_1) & ----- & cov(x_m,x_1) \\
 | & --|-- & | \\
cov(x_m,x_1) & ----- & var(x_n)
\end{bmatrix}$$

The covariance matrix gives a measure of the variance between all variables in the data. From this transformation a basis can be found this is done through eigen decomposition this give the eigenvectors and eigenvalues, which are the root of the principal components. Given a  square matrix $\Sigma$ of $A\in \mathbb{R}^{MxN}$ which has $\leq M$ linear independent eigenvectors, due the data $A$ the number of independent eigenvectors is $\leq N$, we can then factor the matrix $\Sigma$ as 

$$ \Sigma = V \Lambda V^-1 $$

where $V$ is a square matrix, whose columns are the eigenvectors of $\Lambda$ and is a matrix with the eigenvalues along the diagonal.[3] The largest eigenvalues and its related eigenvector is principal components one, the descending eigenvalues and its related eigenvector are the remaining principal components. As a special case, for every n × n real symmetric matrix, the decomposed matrix can be written as 

$$ \Sigma = V \Lambda V^T $$

theses eigenvectors form a new ortho basis sometimes called the “variable space”

Another factoring technique is $QR$ this allow for a matrix to be factored into a 
orthonormal matrix $Q$ and $R$ is a right triangular matrix.[3]

$$ A = QR$$

$Q$ is found using a process called Gram-Schmidt. this is calculated by working with the columns of a matrix for example $A$. 

$$ A  = \begin{bmatrix}
a_1 | & a_2 |& ... & | a_n 
\end{bmatrix}$$

from here the these columns are normalized by setting $u_1 = a_1$  and then normalising $u_1$, 

$$e_1 = \frac {u_1}{||u_1||} $$

for $u_2$ we have subtract the project $a_2$ on to $a_1$ this results in  

$$u_2  =  a_2 - proj_{u_1} a_2 $$

and 

$$e_2 = \frac {u_2}{||u_2||} $$

this is subtract of projection process is used for each follow column. this can be written as 

$$ u_n = a_n - \sum_{i=1}^{n-1} proj_{u_i} a_n $$

$$ e_n = \frac {u_n}{||u_n||} $$

these $e_n$ vectors are then stacked into a matrix forming $Q$, 

$$ Q  = \begin{bmatrix}
e_1 | & e_2 |& ... & | e_n 
\end{bmatrix}$$

for $R$ we and simply multiply $A$ and $Q^{-1}$ 


$$ R= A Q^{-1} $$

Lastly, there are some scaling factors to keep in mind when working with images and raw data. The WFPC2 uses a CCD sensor,[4]  these sensors have a linear response to light intensity. This is a problem for the human eye as humans process light intensity on a logarithmic scale. Due to this difference the data will have to be scaled before a human can view the image. [5]

### Dimension Reduction
The collection of data used come from the Hubble Legacy Archive(HLA) the exact files are from the HST  07515-01-wfpc2.[4] This included the files using the following filters F487N, F502N, F547M, F656N, F658N, and F673N. 
this data is from NGC 7635 or more commonly known as the bubble nebula.[4] the problem arises when trying to combine this data into a single image of color or gray scale. For color this a six to three dimension reduction and for gray scale this is a six to one reduction. The method chosen to reduce these dimensions was principal component analysis (PCA). Due to the nature of the image once the principal component's (PC's) are calculated the data cant be plotted in the “variable space” instead the PC must be mapped back to the original space and collapsed down into once images for each PC. To preform the PCA the data was reading using the astropy library for python. The data for each filter was then flatted into a vector $V_n\in \mathbb{R}^{2722500x6}$ these were then stacked forming at matrix $A\in \mathbb{R}^{2722500x6}$. When calculating the covariance matrix, a matrix of 2722500 by 2722500 would be formed. In order to calculate this by taking $AA^T$ the computer would utilize 53.9 TB of memory. This is not a feasible calculation for a standard computer, so another method was derived. Using QR factorization on the data $A$ this results in a  $Q\in \mathbb{R}^{2722500x6}$ and a $R\in \mathbb{R}^{6x6}$ using this we find $AA^T$

$$AA^T = QRQ^TR^T$$

$$AA^T \approx RR^T$$

this come from the fact that $QQ^T$ is approximately the identity matrix. The calculation of $RR^T$ is a feasible calculation for a standard computer as this returns a six by six matrix. From here eigen-decomposition was preformed due to the fact this was a symmetric matrix, the decomposed matrix can be calculated as 

$$ RR^T = V \Lambda V^T $$

this returns six eigenvectors of size six. These value are the thought as weights these weights are mapped back to the original space for all six eigenvectors. 

$$ R_k  = ((V_k \Lambda_k )V_k^T) R^{-T}$$

$$A_k = QR_k$$

$A_k\in \mathbb{R}^{2722500x6}$ is collapsed down by adding all the colunms together returning a $A_k\in \mathbb{R}^{2722500x1}$ this can then be inflated back into a 1650 by 1650 photo.

### Results
When preforming the PCA due to the nature of the sensor the data is skewed see Figure 1. The data outside the desired values were zeros out before the PCA is performed. As described above PCA was preformed on the data and the eigenvalues were plotted on a scree plot shown in Figure 2. The total amount of data for the first PC was calculated and found to be $83.1 \% $ and the first three PC's accounted for $95.3 \% $
After mapping back to the original space a histogram plot of the data reveals there are negative values for each PC. See figure 3. To scale the data and remove negative values the date is squared. Due to the nature of the human eye the natural logarithm is applied. The Data is then normalized. see Figure 1. To build a colored photo the first three principal components are mapped back original space and the scaling is applied. An array of zeros is set up of the original size with 3 channels blue green and red. $A_1$ is assigned to the green channel $A_2$ to the blue channel and $A_3$ to red channel. See figure 4. The color process was also performed on $A_4$, $A_5$, $A_6$.see Figure 5. the last three $A_k$ shows the data that is most related. Where the first three shows the data that has the most variance 
<p align="center">
 <img width="50%" height="50%" src="https://github.com/MathandDataScience/Dimension-Reduction-PCA/blob/main/Pictures/PC1.png" width="50%" height="50%">
</p>
<p align="center">
 <em>Figure 1: WFPC2 bubble nebula: Image created by PC1</em>
</p>

<p align="center">
 <img width="50%" height="50%" src="https://github.com/MathandDataScience/Dimension-Reduction-PCA/blob/main/Pictures/screeplot.png" width="50%" height="50%">
</p>
<p align="center">
 <em>Figure 2: Scree plot of the eigenvalues</em>
</p>

<p align="center">
 <img width="50%" height="50%" src="https://github.com/MathandDataScience/Dimension-Reduction-PCA/blob/main/Pictures/PC1-.png" width="50%" height="50%">
</p>
<p align="center">
 <em>Figure 3: Histogram of the mapped back data of PC1</em>
</p>

<p align="center">
 <img width="50%" height="50%" src="https://github.com/MathandDataScience/Dimension-Reduction-PCA/blob/main/Pictures/PC1-3.png" width="50%" height="50%">
</p>
<p align="center">
 <em>Figure 4: WFPC2 bubble nebula: Image created by PC1-3</em>
</p>

<p align="center">
 <img width="50%" height="50%" src="https://github.com/MathandDataScience/Dimension-Reduction-PCA/blob/main/Pictures/PC4-6.png" width="50%" height="50%">
</p>
<p align="center">
 <em>Figure 5: WFPC2 bubble nebula: Image created by PC4-6</em>
</p>


### Conclusion

In summary when working with large sample size data set and PCA is believed to offer some insight into the data it may be helpful to use a QR factorization approach to PCA. At a minimum this will cut down on computation time.[3] From our data set we were able to maintain  83.1 %  of the information in just one photo graph using PCA. When working with 3 channels we can maintain  95.3 %  of the information. Lastly, I would like to express sincere gratitude to my project supervisors, Dr. Seth Dutter and Dr.Keith Wojciechowski. 


### References

[1] Gonzalez, Rafael C., Woods, Richard E. (2008).Digitalimageprocessing (3rd d.). Pren-
tice Hall

[2] Abdi, H., and Williams, L. J. (2010). Principal component analysis. Wiley Inter-
disciplinary Reviews. Computational Statistics , 2(4), 433–459. [online] Available at:
https://doi.org/10.1002/wics.101

[3] Sharma, A., Paliwal, K.K., Imoto, S. (2013).,P rincipalcomponentanalysisusingQRdecomposition.
Int Journal of Machine Learning Cybernetics 4, 679–683. [online] Available at:
https://doi.org/10.1007/s13042-012-0131-7

[4] hst 07515 01 wfpc2 files (2023). [online] Available at: https://hla.stsci.edu/ [Ac- cessed
20 May 2023].

[5] Kenneth R.Spring, T. J. Fellers, M. W. Davidson, HumanV isionandColorP erception.
[online] Available at: https://www.olympus-lifescience.com/en/microscope-
resource/primer/lightandcolor/humanvisionintro/



