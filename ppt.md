
<style>
.small-code pre code {
  font-size: 1em;
}


ppt
========================================================
author: Vishal Shukla
date: 
autosize: true


========================================================
class: small-code
Save data as csv as it is way faster than xlsx. 

- Check which records dont have pollarity assigned and take care of it.



```r
which(is.na(twt$polarity))
```

- Check out class distribution

```r
table(twt$polarity)
```

```

  -1    0    1 
7268 2490 2029 
```

Doc Term Matrix
========================================================
class: small-code

```r
traintermmatrix
```

```
<<DocumentTermMatrix (documents: 8840, terms: 12899)>>
Non-/sparse entries: 86071/113941089
Sparsity           : 100%
Maximal term length: 324
Weighting          : term frequency (tf)
```
We cut down the matrix drastically by removing sparse terms.

```r
traintermmatrix <- removeSparseTerms(traintermmatrix, 0.99)
traintermmatrix
```

```
<<DocumentTermMatrix (documents: 8840, terms: 161)>>
Non-/sparse entries: 39460/1383780
Sparsity           : 97%
Maximal term length: 15
Weighting          : term frequency (tf)
```
From 12930 terms to 163 terms

Dictionary reduction methods
========================================================
- Use local dictionary (becomes bag of words model)
- Remove stopwords
- Feature selection
- Token reduction (by stemming, lemmatization, merging synomyms) 

Class imbalance problem
=======================================================
class: small-code
More than 60% of tweets are negative causing significant class imbalance which can have adverse effect on modelling.

The issue can be addressed in the following ways:
- Collect more data (we should not go for it at this stage)
- Upsample / Downsample the imbalanced classes
- Cost sensitive training
  + can specify costs for missclassified minority classes as shown below
  

```r
# In case of C5.0
costMatrix <- matrix(c(0,5,5,10,0,7,6,5,0), nrow = 3)
c50 <- C5.0(as.matrix(traintermmatrix), as.factor(train$polarity), costs = costMatrix)
```

Bigrams
========================================================
class: small-code

```r
library(RWeka)
BigramTokenizer <- function(x) NGramTokenizer(x, Weka_control(min = 2, max = 2))
BigramMatrix <- TermDocumentMatrix(corp, control = list(tokenize = BigramTokenizer))
```
Top five bigram pairs

```
  word pairs frequency
1     i have       339
2  thank you       284
3 my account       232
4    for the       219
5      to be       217
```

Correlation Plot
========================================================
class: small-code
We don't find any major patch of a color shade indicating correlation

```r
library(corrplot)
c <- cor(as.matrix(traintermmatrix))
corrplot(c, method = "color")
```

<img src="ppt-figure/unnamed-chunk-9-1.png" title="plot of chunk unnamed-chunk-9" alt="plot of chunk unnamed-chunk-9" width="380px" height="380px" style="display: block; margin: auto;" />

==========================================================
class: small-code
We can remove correlated terms from the train term matrix as below

```r
library(caret)
correlated = findCorrelation(c, cutoff=0.8) # Find term pairs with more than 80% correlation
termmat <- as.matrix(traintermmatrix)
termmat <- termmat[ , -c(correlated)] # Removing the correlated terms
```

LDA (Latent Dirichlet Allocation)
========================================================
class: small-code
Remove empty documents from the corpus first before applying LDA

```r
library(topicmodels)
lda <- LDA(traintermmatrix, k=10)
```




```r
terms(lda)
```

```
      Topic 1       Topic 2       Topic 3       Topic 4       Topic 5 
     "hsbcuk"        "bank"        "bank"      "hsbcuk"  "hsbcukhelp" 
      Topic 6       Topic 7       Topic 8       Topic 9      Topic 10 
        "amp"    "bofahelp"      "hsbcuk"      "hsbcuk" "hdfcbankcar" 
```
Can remove the handle names from corpus if not wanted as topics

Multi Dimensional Scaling (MDS) with LSA (Latent Symantic Analysis)
===================================================================
class: small-code
Words with close meanings occur in similar kind of documents. In this way LSA analyzes for similarity between the records.

```r
library(lsa)
traintermmatrix.lsa <- lw_bintf(as.matrix(traintermmatrix.new)) * gw_idf(as.matrix(traintermmatrix.new))  
lsaSpace <- lsa(traintermmatrix.lsa)  
dist.mat.lsa <- dist(t(as.textmatrix(lsaSpace)))
fit <- cmdscale(dist.mat.lsa, eig = TRUE, k = 3)
```

===================================================================
class: small-code

```r
library(scatterplot3d)
scatterplot3d(fit$points[, 1], fit$points[, 2], fit$points[, 3], pch = 16, main = "Semantic Space Scaled to 3D", xlab = "x", ylab = "y", zlab = "z", type = "p")
```

<img src="ppt-figure/unnamed-chunk-15-1.png" title="plot of chunk unnamed-chunk-15" alt="plot of chunk unnamed-chunk-15" width="400px" height="400px" style="display: block; margin: auto;" />

PCA (Principal Component Analysis)
==================================================
class: small-code

```r
x <- as.matrix(traintermmatrix.new)
F <- x/rowSums(x)
classpca <- prcomp(F, scale. = T)
biplot(classpca, scale = T)
```

<img src="ppt-figure/unnamed-chunk-16-1.png" title="plot of chunk unnamed-chunk-16" alt="plot of chunk unnamed-chunk-16" style="display: block; margin: auto;" />

SIMCA (Soft Independent Modelling of Class Analogies)
=====================================================
class: small-code

```r
library(mdatools)
sim_neg <- simca(as.matrix(traintermmatrix), "-1")
sim_pos <- simca(as.matrix(traintermmatrix), "1")
sim_nut <- simca(as.matrix(traintermmatrix), "0")
simmulti <- simcam(list(sim_neg, sim_pos, sim_nut))

res_simca <- predict(simmulti, as.matrix(testtermmatrix))
```

SIMCA, a method for supervised classification uses PCA for retaining significant components from the data.


Using Clustering for Classification
==================================================
class: small-code


```r
library(cluster)
d <- dist(t(traintermmatrix), method="euclidian")
kfit <- kmeans(d, 3)
clusplot(as.matrix(d), kfit$cluster, color=T, shade=T, labels=3, lines=0)
```

<img src="ppt-figure/unnamed-chunk-18-1.png" title="plot of chunk unnamed-chunk-18" alt="plot of chunk unnamed-chunk-18" width="425px" height="425px" style="display: block; margin: auto;" />

=================================================
This is a simple cluster formed by kmeans. 

There are other ways with which we can apply clustering to new-data / test set. (Mclust package, etc.)

Although when we know the class of a record, it is a classification problem not clustering. But we can apply clustering anyway and check out the performance.


## Scope for improving performance
===================================================
We can improve model performance by implementing one or many of the following:
- Improve quality of data (Already done by removing irrelevant tweets)
- Add more features to the original data
  + Can add columns for subjectivity/objectivity
  + Can go for varying degrees of sentiments (like -5 to +5 instead of -1 to +1). This can increase the number of classes and hence reduce class imbalance.
- Feature engineering 
- Algorithm parameter tuning
  
None of the above method can single-handedly give miraculous rise in performance. But combination of all best practices can surely benefit.
