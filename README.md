# KiesKompas_PCA
The "KiesKompas" is a favorite voting aid in Dutch electoral seasons. Based on 30 theses, political parties are stratified along two axes: "left/right" and "conservative/progressive". A voter gives his or her opinion on 30 theses, which places him or her somewhere in this space and the proximity to the different political parties then translates into a voting advise.

In the Netherlands parties are however very often distributed along the diagonal that runs from left and progressive down to right and conservative, with very little in the remaining quadrants (right and progressive and left and conservative, respectively). 

To differentiate between the parties it might therefore be more interesting to do a simple Principal Component Analysis (PCA). PCA is a way of reducing the dimensions of a high dimensional dataset while preserving (as much as possible) correlations present in the higher dimensions.

In this case, we work with a 30 dimensional space: 1 dimension for each thesis. The answers to each question can be anything from "very much agree", through "agree", "neutral" and "disagree", to "very much disagree". We will translate this into numerical values: 2, 1, 0, -1 and -2. 

With PCA, we reduce this 30 dimensional space to a 2 dimensional space, which is shown below. 


![PCA of dutch political parties](https://github.com/Josha91/KiesKompas_PCA/blob/main/kieskompas_PCA.png)

For comparison, the 'classic' KiesKompas looks like this for the 2021 elections:

![Classic KiesKompas](https://github.com/Josha91/KiesKompas_PCA/blob/main/kieskompas_original.png)

There are two other glaring issues with the KiesKompas. First, it ignores new parties as long as they don't reach 1 seat in the polls (almost 1% of the electorate), even though the effect of these voting aids can be significant (see http://griffiers.nl/files/vvgtest_csshtml_nl/4.%20Effecten%20van%20online%20stemhulpen.pdf)

The other is a centricity bias. There are many ways to end up in the center of the spectrum. In fact, you can vote completely opposite to a central party and still be advised to vote for them. 
Extreme parties, on the other hand, can only be reached in one way (in the most extreme case): voting in one direction and one direction only. Low-entropy configurations, in essence. 

The code in this repo also explores this. It does this by taking the PCA from above, and applying an inverse transform on it. The inverse transform is then compared to the input (=the votes of the parties on the various topics). The standard deviation of this residual is a measure of how well we can project the answers (a 30 dimensional vector) from the principal components (a 2d vector). We can plot this against a measure of the centricity (in this case, the length of the vector in (PC1, PC2) space. 

What we then see is that indeed the most central parties (lowest centricity) are the least predictable (largest spread in residuals ~ highest entropy). 


![Centricity vs entropy](https://github.com/Josha91/KiesKompas_PCA/blob/main/centricity_vs_inverse_pca_residuals.png)
