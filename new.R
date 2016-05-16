require(tm)
require(caTools)
require(randomForest)
require(RWeka)
#library(DMwR)
require(stringr)
require(stringi)
require(NLP)
require(openNLP)

# Read the xls file and save it as csv. csv is faster.
twt <- read.xlsx("Twitter_v2.xlsx", sheetIndex = 1)
write.csv(twt, file = "twitter.csv")

twt <- read.csv("twitter.csv")

# Finding which records are missed while scoring polarity
which(is.na(twt$polarity))

twt[9422, ]$polarity  <- -1
twt[10269, ]$polarity  <- 1

# Checking out class distribution
table(twt$polarity)


##################################### NER

word_ann <- Maxent_Word_Token_Annotator()
sent_ann <- Maxent_Sent_Token_Annotator()
annotation <- annotate(twt$text, list(sent_ann, word_ann))
org_ann <- Maxent_Entity_Annotator(kind = "organization", language = "en", probs = F, model = NULL)

##########################################

## Before removing twitter handles ############################################
twt$text <- str_replace_all(twt$text, "#", " ")
twt$text <- str_replace_all(twt$text, "@", " ")
twt$text <- tolower(twt$text)

# Normalizing text slightly improves acc
twt$text <- stri_trans_nfc(twt$text)

# sampling and spliting data into training and test set
split <- sample.split(twt$text, SplitRatio = 0.75)
train <- subset(twt, split == T)
test <- subset(twt, split == F)

toSpace <- content_transformer(function(x, pattern) gsub(pattern, " ", x))

# Processing training data
trainvector <- as.vector(train$text)
corp <- Corpus(VectorSource(trainvector))

corp <- tm_map(corp, toSpace, "@")
corp <- tm_map(corp, toSpace, "#")
corp <- tm_map(corp, toSpace, "$")
corp <- tm_map(corp, content_transformer(tolower))
corp <- tm_map(corp, stemDocument, language = "english")
corp <- tm_map(corp, removeNumbers)
corp <- tm_map(corp, removePunctuation)
corp <- tm_map(corp, stripWhitespace)

traintermmatrix <- DocumentTermMatrix(corp, control = list(weighting = weightTf, stopwords = T))
traintermmatrix <- removeSparseTerms(traintermmatrix, 0.9997)
traintermmatrix <- removeSparseTerms(traintermmatrix, 0.995)
traintermmatrix <- removeSparseTerms(traintermmatrix, 0.990) #163 terms left

## 0.995 will reduce the matrix around 300 terms

traintermmatrix

# Processing testing data for preparing dtest
testvector <- as.vector(test$text)
corp <- Corpus(VectorSource(testvector))

corp <- tm_map(corp, toSpace, "@")
corp <- tm_map(corp, toSpace, "#")
corp <- tm_map(corp, toSpace, "$")
corp <- tm_map(corp, content_transformer(tolower))
corp <- tm_map(corp, stemDocument, language = "english")
corp <- tm_map(corp, removeNumbers)
corp <- tm_map(corp, removePunctuation)
corp <- tm_map(corp, stripWhitespace)
testtermmatrix <- DocumentTermMatrix(corp, control = list(weighting = weightTf, stopwords = T))

## Correlation analysis
library(corrplot)
c <- cor(as.matrix(traintermmatrix))
corrplot(c, method = "color")

############# Remove correlated terms
library(caret)
hc = findCorrelation(c, cutoff=0.8)
hc = sort(hc)
mat <- as.matrix(traintermmatrix)
mat <- mat[, -c(hc)]
##################################


plot(traintermmatrix, terms=findFreqTerms(traintermmatrix, lowfreq=100)[1:10], corThreshold=0.5)

## Top frequent terms
findFreqTerms(traintermmatrix, lowfreq = 100)
findFreqTerms(traintermmatrix, highfreq = 100)

freq <- colSums(as.matrix(traintermmatrix))
freq <- sort(freq, decreasing = T)
freq <- as.data.frame(freq)
freq <- data.frame(rownames(freq), freq$freq, prop.table(freq))
colnames(freq) <- c("terms", "frequency", "percentage proportion")
x <- subset(freq[1:30,])
ggplot(x, aes(x = reorder(x$terms, -x$frequency), y = x$frequency)) + geom_bar(stat = "Identity", fill = "Sky Blue") + ggtitle("Top 30 frequent terms") + xlab("Terms") + theme(axis.text.x = element_text(angle = 45, hjust = 1))

freqlowtohigh <- colSums(as.matrix(traintermmatrix))
freqlowtohigh <- sort(freqlowtohigh, decreasing = F)
freqlowtohigh <- as.data.frame(freqlowtohigh)
freqlowtohigh <- data.frame(rownames(freqlowtohigh), freqlowtohigh$freq, prop.table(freqlowtohigh))
colnames(freq) <- c("terms", "frequency", "percentage proportion")
x <- subset(freqlowtohigh[1:30,])


##Bigrams
BigramTokenizer <- function(x) NGramTokenizer(x, Weka_control(min = 2, max = 2))
BigramMatrix <- TermDocumentMatrix(corp, control = list(tokenize = BigramTokenizer))
BigramMatrix <- removeSparseTerms(BigramMatrix, 0.998)

bitestmat <- TermDocumentMatrix(corp, control = list(tokenize = BigramTokenizer))
bitestmat <- removeSparseTerms(bitestmat, 0.998)

freqBi <- rowSums(as.matrix(BigramMatrix))
freqBi <- sort(freqBi, decreasing = TRUE)
freqBi <- as.data.frame(freqBi)
freqBi <- data.frame(rownames(freqBi), freqBi$freqBi)
colnames(freqBi) <- c("terms", "frequency")
x <- subset(freqBi[1:30,])
ggplot(x, aes(x = reorder(x$terms, -x$frequency), y = x$frequency)) + geom_bar(stat = "Identity", fill = "Sky Blue") + ggtitle("Top 30 frequent terms") + xlab("Terms") + theme(axis.text.x = element_text(angle = 45, hjust = 1))

## RF model along with tuned mtry
t <- tuneRF(as.matrix(traintermmatrix), as.factor(train$polarity), stepFactor = 0.5, plot = T)
model_rf <- randomForest(as.matrix(traintermmatrix), as.factor(train$polarity))
results_rf <- predict(model_rf, as.matrix(testtermmatrix))
table(results_rf, test$polarity)

# Set proximity = T in rF to use it for MDS plot
MDSplot(model_rf, as.factor(train$polarity), k=3)

## SVM
## Terms/dictionary for train and test must be same to apply svm
library(e1071)
model_svm <- svm(as.matrix(traintermmatrix), as.factor(train$polarity))
features <- traintermmatrix$dimnames$Terms
testtermmatrix <- DocumentTermMatrix(corp, control = list(weighting = weightTf, stopwords = F, dictionary = features))
result_svm <- predict(model_svm, as.matrix(testtermmatrix))
table(result_svm, test$polarity)

######################## Using C5.0 ########################
library(C50)
cont <- C5.0Control(subset = T, bands = 0, winnow = T, noGlobalPruning = T, CF = 0.25, minCases = 2, fuzzyThreshold = T)
c50 <- C5.0(as.matrix(traintermmatrix), as.factor(train$polarity), control =  cont)
result_c50 <- predict.C5.0(c50, newdata = as.matrix(testtermmatrix), type = "class")
table(result_c50, test$polarity)

## Cost sensitive training in c5.0
costs <- matrix(c(1,5,5,10,0,7,6,5,0), nrow = 3)
c50 <- C5.0(as.matrix(traintermmatrix), as.factor(train$polarity), control =  cont, costs = costs)


## Trying PCA
## Remove empty documents first
x <- as.matrix(traintermmatrix.new)
F <- x/rowSums(x)
classpca <- prcomp(F, scale. = T)
plot(classpca)
biplot(classpca, scale = T)
# biplot(classpca, expand = 5, scale = T, xlim = C(-0.050, 0.000), ylim = c(0,20))

####### Kernel PCA
library(kernlab)
kp <- kpca(as.matrix(traintermmatrix))

## Trying LDA (Latent Dirichlet allocation)
## LDA needs dtm with TF weightng
## Also remove empty documents from dtm
library(topicmodels)
traintermmatrix <- DocumentTermMatrix(corp, control = list(weighting = weightTf, stopwords = T))
traintermmatrix <- removeSparseTerms(traintermmatrix, 0.98)

BigramMatrix <- TermDocumentMatrix(corp, control = list(tokenize = BigramTokenizer, weighting = weightTf))
BigramMatrix <- removeSparseTerms(BigramMatrix, 0.998)
# Remove empty documents from dtm
rowTotals <- apply(as.matrix(traintermmatrix), 1, sum)
traintermmatrix.new   <- traintermmatrix[rowTotals> 0, ]

lda <- LDA(traintermmatrix.new, k=10)

terms(lda)
topics(lda)

ldabi <- LDA(BigramMatrix, k=10)
terms(ldabi)
topics(ldabi)


## Modelling on Bigrams
nb <- naiveBayes(t(as.matrix(BigramMatrix)), train$polarity)
rnb <- predict(nb, t(as.matrix(bitestmat)))
table(rnb, test$polarity)

########
td.mat <- as.matrix(traintermmatrix.new)
dist.mat <- dist(t(as.matrix(td.mat)))
dist.mat

mds <- cmdscale(dist.mat, eig = T, k = 3)
pts <- data.frame(x = mds$points[,1], y = mds$points[,2])
ggplot(pts, aes(x = x, y = y)) + geom_point(data = pts, aes(x = x, y = y, color = df$view)) + geom_text(data = pts, aes(x = x, y = y - 0.2, label = row.names(df)))


#######################3
library(lsa)
td.mat.lsa <- lw_bintf(td.mat) * gw_idf(td.mat)  # weighting
lsaSpace <- lsa(td.mat.lsa)  # create LSA space
dist.mat.lsa <- dist(t(as.textmatrix(lsaSpace)))
fit <- cmdscale(dist.mat.lsa, eig = TRUE, k = 3)
points <- data.frame(x = fit$points[, 1], y = fit$points[, 2])
ggplot(points, aes(x = x, y = y)) + geom_point(data = points, aes(x = x, y = y, color = View(df))) + geom_text(data = points, aes(x = x, y = y - 0.2, label = row.names(df)))

library(scatterplot3d)
fit <- cmdscale(dist.mat.lsa, eig = TRUE, k = 3)
#colors <- rep(c("blue", "green", "red"), each = 3)
scatterplot3d(fit$points[, 1], fit$points[, 2], fit$points[, 3], pch = 16, main = "Semantic Space Scaled to 3D", xlab = "x", ylab = "y", zlab = "z", type = "h")

#### Clustering by term similarity 
library(cluster)   
d <- dist(t(traintermmatrix), method="euclidian")
fit <- hclust(d=d, method="ward.D") 
fit
plot(fit, hang=-1) 
cutree(fit, k=3)
rect.hclust(fit, k=3, border="red")
## Kmeans
library(fpc)
kfit <- kmeans(d, 3)
clusplot(as.matrix(d), kfit$cluster, color=T, shade=T, labels=3, lines=0)


###########
d2 <- dist(t(testtermmatrix), method="euclidian")
clusplot(as.matrix(d2), kfit$cluster, color=T, shade=T, labels=3, lines=0)

########## MClust
library(mclust)
model <- MclustDA(as.matrix(traintermmatrix), as.factor(train$polarity))


###############
library(mdatools)
sim <- simca(as.matrix(traintermmatrix), "-1")
simpos <- simca(as.matrix(traintermmatrix), "1")
simnu <- simca(as.matrix(traintermmatrix), "0")

simmulti <- simcam(list(sim, simpos, simnu))
plot(simmulti)

features <- traintermmatrix$dimnames$Terms
testtermmatrix <- DocumentTermMatrix(corp, control = list(weighting = weightTf, stopwords = T, dictionary = features))

res_sim <- predict(simmulti, as.matrix(testtermmatrix))

########## Bag of Words + SVM ##########
features <- c('wtf', 'wait', 'waiting', 'fail', 'support', 'dont', 'doesnt', 'arent', 'isnt', 'didnt', 'havent', 'hadnt', 'shouldnt', 'wouldnt', 'wont', 'not', 'cannot', 'couldnt', 'cant', 'fuck', 'worst', 'shit', 'yall', 'thank', 'thanks', 'hold', 'app', 'service', 'issues', 'problems', 'damn', 'lost', 'still', 'please', 'money', 'out', 'for', 'you', 'proud', 'congrats', 'best', 'great', 'appreciate', 'award', 'just', 'email', 'send', 'call', 'need', 'credit', 'debit', 'card', 'cash', 'banking', 'paid', 'people', 'pay', 'waiting', 'time', 'help', 'issue', 'problem', 'im', 'fix', 'fee', 'account')


trainvector <- as.vector(train$text)
corp <- Corpus(VectorSource(trainvector))

corp <- tm_map(corp, toSpace, "@")
corp <- tm_map(corp, toSpace, "#")
corp <- tm_map(corp, toSpace, "$")
corp <- tm_map(corp, content_transformer(tolower))
corp <- tm_map(corp, stemDocument, language = "english")
corp <- tm_map(corp, removeNumbers)
corp <- tm_map(corp, removePunctuation)
corp <- tm_map(corp, stripWhitespace)

testvector <- as.vector(test$text)
corp <- Corpus(VectorSource(testvector))

corp <- tm_map(corp, toSpace, "@")
corp <- tm_map(corp, toSpace, "#")
corp <- tm_map(corp, toSpace, "$")
corp <- tm_map(corp, content_transformer(tolower))
corp <- tm_map(corp, stemDocument, language = "english")
corp <- tm_map(corp, removeNumbers)
corp <- tm_map(corp, removePunctuation)
corp <- tm_map(corp, stripWhitespace)


traintermmatrix <- DocumentTermMatrix(corp, control = list(weighting = weightTf, stopwords = F, dictionary = features))
testtermmatrix <- DocumentTermMatrix(corp, control = list(weighting = weightTf, stopwords = F, dictionary = features))

model_svm <- svm(as.matrix(traintermmatrix), as.factor(train$polarity))
result_svm <- predict(model_svm, as.matrix(testtermmatrix))
table(result_svm, test$polarity)

############ Analyze words from different classes
neg <- subset(twt, twt$polarity == "-1")
pos <- subset(twt, twt$polarity == "1")
neut <- subset(twt, twt$polarity == "0")

neg_vector <- as.vector(neg$text)
pos_vector <- as.vector(pos$text)
neut_vector <- as.vector(neut$text)

c <- Corpus(VectorSource(neg_vector))
c <- tm_map(c, toSpace, "@")
c <- tm_map(c, toSpace, "#")
c <- tm_map(c, toSpace, "$")
c <- tm_map(c, content_transformer(tolower))
c <- tm_map(c, removeNumbers)
c <- tm_map(c, removePunctuation)
c <- tm_map(c, stripWhitespace)
negtermmat <- DocumentTermMatrix(c, control = list(weighting = weightTf, stopwords = F))

c <- Corpus(VectorSource(pos_vector))
c <- tm_map(c, toSpace, "@")
c <- tm_map(c, toSpace, "#")
c <- tm_map(c, toSpace, "$")
c <- tm_map(c, content_transformer(tolower))
c <- tm_map(c, removeNumbers)
c <- tm_map(c, removePunctuation)
c <- tm_map(c, stripWhitespace)
postermmat <- DocumentTermMatrix(c, control = list(weighting = weightTf, stopwords = F))

c <- Corpus(VectorSource(neut_vector))
c <- tm_map(c, toSpace, "@")
c <- tm_map(c, toSpace, "#")
c <- tm_map(c, toSpace, "$")
c <- tm_map(c, content_transformer(tolower))
c <- tm_map(c, removeNumbers)
c <- tm_map(c, removePunctuation)
c <- tm_map(c, stripWhitespace)
neuttermmat <- DocumentTermMatrix(c, control = list(weighting = weightTf, stopwords = F))


freq <- colSums(as.matrix(negtermmat))
freq <- colSums(as.matrix(postermmat))
freq <- colSums(as.matrix(neuttermmat))


freq <- sort(freq, decreasing = T)
freq <- as.data.frame(freq)
freq <- data.frame(rownames(freq), freq$freq, prop.table(freq))
colnames(freq) <- c("terms", "frequency", "percentage proportion")
x <- subset(freq[1:100,])
ggplot(x, aes(x = reorder(x$terms, -x$frequency), y = x$frequency)) + geom_bar(stat = "Identity", fill = "Sky Blue") + ggtitle("Top 30 frequent terms") + xlab("Terms") + theme(axis.text.x = element_text(angle = 45, hjust = 1))

