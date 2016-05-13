# Load required packages

require(xlsx)
require(tm)
require(randomForest)
require(caret)
require(e1071)
require(ggplot2)
require(wordcloud)
require(SnowballC)
require(xgboost)

# Load the data into the memory

twt <- read.xlsx("bank_tweets_scored.xlsx", sheetIndex = 1)
trainvector <- as.vector(twt$text)

# Creating text corpus using tweets

corp <- Corpus(VectorSource(trainvector))

# Function to transform input pattern into space

toSpace <- content_transformer(function(x, pattern) gsub(pattern, " ", x))

# Data preprocessing steps

## Removing hashtags(#), mentions(@) and other special characters from corpus
corp <- tm_map(corp, toSpace, "@")
corp <- tm_map(corp, toSpace, "#")
corp <- tm_map(corp, toSpace, "$")

## changing all text to lowercase
corp <- tm_map(corp, content_transformer(tolower))

## stemming 
corp <- tm_map(corp, stemDocument, language = "english")

## Removing numbers and punctuation
corp <- tm_map(corp, removeNumbers)
corp <- tm_map(corp, removePunctuation)

## Removing erroneous whitespaces
corp <- tm_map(corp, stripWhitespace)

## Creating document term matrix and taking steps like
## tf-idf weighting of terms and removal of stopwords
termmatrix <- DocumentTermMatrix(corp, control = list(weighting = weightTfIdf, stopwords = T))
termmatrix <- removeSparseTerms(termmatrix, 0.98)

# Preparing a test model using random forest and support vector machine
model_rf <- randomForest(as.matrix(termmatrix), as.factor(twt$polarity))
model_svm <- svm(as.matrix(termmatrix), as.factor(twt$polarity), kernel = "radial")
model_NB <- naiveBayes(as.matrix(termmatrix), as.factor(twt$polarity))

# Predicting results (on training set itself) based on the model
results_rf <- predict(model_rf, as.matrix(termmatrix))
results_svm <- predict(model_svm, as.matrix(termmatrix))
results_NB <- predict(model_NB, as.matrix(termmatrix))

# Viewing performance using classification table (actual vs predicted)
table(results_rf, twt$polarity)
table(results_svm, twt$polarity)
table(results_NB, twt$polarity)

freq <- colSums(as.matrix(termmatrix))
freq <- sort(freq, decreasing = T)
#head(freq, 20)

freq <- as.data.frame(freq)
freq <- data.frame(rownames(freq), freq$freq, prop.table(freq))
colnames(freq) <- c("terms", "frequency", "percentage proportion")

x <- subset(freq[1:30,])
ggplot(x, aes(x = reorder(x$terms, -x$frequency), y = x$frequency)) + geom_bar(stat = "Identity", fill = "Sky Blue") + ggtitle("Top 30 frequent terms") + xlab("Terms") + theme(axis.text.x = element_text(angle = 45, hjust = 1))

wordcloud(freq$terms, freq$frequency, max.words = 100, rot.per = 0.2, colors = brewer.pal(8, "Dark2"))

f <- function(x){if (x == -1) return(2) else return(x)}
pol <- lapply(twt$polarity, f)
pol <- as.numeric(pol)

dtrain <- xgb.DMatrix(as.matrix(termmatrix), label = pol)
watch <- list(train = dtrain, valid = dtrain)
mboost <- xgb.train(data = dtrain, eta = 2.1, objective = "multi:softmax", num_class = 3, colsample_bytree = 0.75, min_child_weight = 1, max_depth = 15, watchlist = watch, nrounds = 100, eval_metric = "mlogloss")
mboost <- xgb.train(data = dtrain, eta = 0.001, objective = "multi:softmax", num_class = 3, min_child_weight = 1, max_depth = 15, watchlist = watch, nrounds = 100, eval_metric = "merror")
pred <- predict(mboost, dtrain)

table(pol, pred)

#90.96% - Max then it levels off
mboost <- xgb.train(data = dtrain, eta = 0.4, objective = "multi:softmax", num_class = 3, colsample_bytree = 0.75, min_child_weight = 1, max_depth = 15, watchlist = watch, nrounds = 2200, eval_metric = "mlogloss")


################ Removing twitter handle terms from corpus #####
stopwordlist <- c(stopwords("english"), "rt", "amp")
stopwordlist <- setdiff(stopwordlist, "not")

corp <- tm_map(corp, removeWords, stopwordlist)

termmatrix <- DocumentTermMatrix(corp, control = list(weighting = weightTf))

######################## Using C5.0 ########################
library(C50)
cont <- C5.0Control(subset = T, bands = 0, winnow = T, noGlobalPruning = T, CF = 0.25, minCases = 2, fuzzyThreshold = T)
c <- C5.0(as.matrix(traintermmatrix), as.factor(train$polarity), control =  cont)
result_c50 <- predict(c, newdata = as.matrix(testtermmatrix), type = "class")
table(result_c50, test$polarity)

######################### RF ##############################
t <- tuneRF(as.matrix(traintermmatrix), as.factor(train$polarity), stepFactor = 0.5, plot = T)
model_rf <- randomForest(as.matrix(traintermmatrix), as.factor(train$polarity), mtry = 8, ntree = 150)
results_rf <- predict(model_rf, as.matrix(testtermmatrix))
table(results_rf, test$polarity)

########################## SVM ##############################
model_svm <- svm(as.matrix(traintermmatrix), as.factor(train$polarity))
features <- traintermmatrix$dimnames$Terms
testtermmatrix <- DocumentTermMatrix(corp, control = list(weighting = weightTf, stopwords = F, dictionary = features))
result_svm <- predict(model_svm, as.matrix(testtermmatrix))
table(result_svm, test$polarity)
