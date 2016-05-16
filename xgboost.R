# Load required packages
require(xlsx)
require(caTools)
require(tm)
require(xgboost)

# Load the data into the memory
twt <- read.xlsx("bank_tweets_scored.xlsx", sheetIndex = 1)
twt <- read.csv("bank_tweets_scored.csv")

# Dividing data for hold-out approach
split <- sample.split(twt$text, SplitRatio = 0.75)
train <- subset(twt, split == T)
test <- subset(twt, split == F)

# Function to transform input pattern into space
toSpace <- content_transformer(function(x, pattern) gsub(pattern, " ", x))

# Processing the training data for dtrain
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
traintermmatrix <- DocumentTermMatrix(corp, control = list(weighting = weightTfIdf, stopwords = F))
traintermmatrix <- removeSparseTerms(traintermmatrix, 0.98)


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
testtermmatrix <- DocumentTermMatrix(corp, control = list(weighting = weightTfIdf, stopwords = F))
# testtermmatrix <- removeSparseTerms(testtermmatrix, 0.98)

# Recoding -1 labels as 2
## As multi:softmax doesnt take -1 as label name
f <- function(x){if (x == -1) return(2) else return(x)}
poltrain <- lapply(train$polarity, f)
poltest <- lapply(test$polarity, f)

# Preparing dtrain and dtest to feed into xgboost
dtrain <- xgb.DMatrix(as.matrix(traintermmatrix), label = poltrain)
dtest <- xgb.DMatrix(as.matrix(testtermmatrix), label = poltest)

watch <- list(train = dtrain, valid = dtest)

poltest <- as.numeric(poltest)

# XGBoost model and prediction on the held out data
mboost <- xgb.train(data = dtrain, eta = 0.002, objective = "multi:softmax", num_class = 3, watchlist = watch, nrounds = 1000, eval_metric = "mlogloss", colsample_bytree = 0.75, min_child_weight = 1, max_depth = 15, subsample = 0.9)
pred <- predict(mboost, dtest)

table(poltest, pred)
