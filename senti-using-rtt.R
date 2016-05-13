require(xlsx)
require(tm)
require(SnowballC)
require(RTextTools)
require(stringr)

memory.size(max = T)

twt <- read.xlsx("bank_tweets_scored.xlsx", sheetIndex = 1)

twt$text <- str_replace_all(twt$text, "#", " ")
twt$text <- str_replace_all(twt$text, "@", " ")
matrix <- create_matrix(twt, language = "english", toLower = T, removeStopwords = T, removeNumbers = T, removePunctuation = T, stemWords = F, stripWhitespace = T, tm::weightTfIdf)
matrix <- removeSparseTerms(matrix, 0.97)

container <- create_container(matrix, twt$polarity, trainSize = 1:12000, testSize = 12001:13924, virgin = F)
model <- train_models(container, algorithms = c("RF", "SVM", "BOOSTING", "BAGGING", "MAXENT", "TREE"))
results <- classify_models(container, model)
analytics <- create_analytics(container, results)
summary(analytics)
