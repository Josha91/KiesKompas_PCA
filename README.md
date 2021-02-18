# KiesKompas_PCA
The "KiesKompas" is a favorite voting aid in Dutch electoral seasons. Based on 30 theses, political parties are stratified along two axes: "left/right" and "conservative/progressive". A voter gives his or her opinion on 30 theses, which places him or her somewhere in this space and the proximity to the different political parties then translates into a voting advise.

In the Netherlands parties are however very often distributed along the diagonal that runs from left and progressive down to right and conservative, with very little in the remaining quadrants (right and progressive and left and conservative, respectively). 

To differentiate between the parties it might therefore be more interesting to do a simple Principal Component Analysis (PCA). PCA is a way of reducing the dimensions of a high dimensional dataset while preserving (as much as possible) correlations present in the higher dimensions.

In this case, we work with a 30 dimensional space: 1 dimension for each thesis. The answers to each question can be anything from "very much agree", through "agree", "neutral" and "disagree", to "very much disagree". We will translate this into numerical values: 2, 1, 0, -1 and -2. 

With PCA, we reduce this 30 dimensional space to a 2 dimensional space, which is shown below. 


![PCA of dutch political parties](https://github.com/Josha91/KiesKompas_PCA/blob/main/kieskompas_PCA.png)
