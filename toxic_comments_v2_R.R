# toxic comments
# updated 10/9/20

# ----------------------------------------------------------------------------
# project setup
# ----------------------------------------------------------------------------

rm(list = ls())

setwd("C:/Users/Walter.Guillioli/Documents/GitHub/toxic-comments/")

require(quanteda)
require(quanteda.textmodels)
require(tidytext)
require(tidyverse)
require(glmnet)
require(caret)
require(e1071)
require(doParallel)


# ----------------------------------------------------------------------------
# load data provided by kaggle
# ----------------------------------------------------------------------------

train <- read.csv("./data/train.csv", header = TRUE, stringsAsFactors = FALSE)

test <- read.csv("./data/test.csv", header = TRUE, stringsAsFactors = FALSE)

test_labels <- read.csv("./data/test_labels.csv", header = TRUE, stringsAsFactors = FALSE)


# ----------------------------------------------------------------------------
# basic data preparation
# ----------------------------------------------------------------------------

# append labels to test df
test <- merge(test, test_labels)

# combine train and test set dataset
train$group <- "train"
test$group <- "test"
comments_df <- rbind(train, test)

target_variables <- c("toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate")


# ----------------------------------------------------------------------------
# create corpus and document feature matrix
# ----------------------------------------------------------------------------

# train corpus - create and explore
train_corpus <- corpus(train,
                       text_field = "comment_text",
                       docid_field	 = "id")

summary(train_corpus,10)
docvars(train_corpus)[1:10,]
docnames(train_corpus)[1:25]
ndoc(train_corpus) 

# test corpus - create and explore
test_corpus <- corpus(test,
                       text_field = "comment_text",
                       docid_field	 = "id")

summary(test_corpus,10)

docvars(test_corpus)[1:10,]

docnames(test_corpus)[1:25]

ndoc(test_corpus) 

# train dfm - create and explore
train_dfm <- dfm(train_corpus,
                 tolower = TRUE,
                 stem = TRUE,
                 remove = stopwords("english"),
                 remove_punct = TRUE,
                 remove_symbols = TRUE,
                 remove_numbers = TRUE,
                 remove_separators = TRUE,
                 verbose = TRUE)

dim(train_dfm)

topfeatures(train_dfm, 50)

train_dfm[1:10, 1:10]

# test dfm - create and explore
test_dfm <- dfm(test_corpus,
                 tolower = TRUE,
                 stem = TRUE,
                 remove = stopwords("english"),
                 remove_punct = TRUE,
                 remove_symbols = TRUE,
                 remove_numbers = TRUE,
                 remove_separators = TRUE,
                 verbose = TRUE)

dim(test_dfm)

topfeatures(test_dfm, 50)

test_dfm[1:10, 1:10]

#DO: trim dfms based on this https://tutorials.quanteda.io/basic-operations/dfm/dfm/
#DO: TFIDF?


# ----------------------------------------------------------------------------
# basic plots - train dfm
# ----------------------------------------------------------------------------

# top 20 tokens 
train_dfm %>% 
  textstat_frequency(n = 20) %>% 
  ggplot(aes(x = reorder(feature, frequency), y = frequency)) +
  geom_point() +
  coord_flip() +
  labs(x = NULL, y = "Frequency") +
  theme_minimal()

# word cloud of top 100 words
set.seed(132)
textplot_wordcloud(train_dfm, max_words = 100)

# word cloud by target variables
for (col in target_variables){
  train_dfm_grouped <- dfm(train_dfm, groups = col)
  print(col)
  set.seed(132)
  textplot_wordcloud(train_dfm_grouped, comparison = TRUE, max_words = 200)
}


# ----------------------------------------------------------------------------
# sentiment data calculate and append
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
# train NB classifiers
# ----------------------------------------------------------------------------

# dfm with matched features of test and train for prediction
matched_dfm <- dfm_match(test_dfm, features = featnames(train_dfm))

#function that gets column to predict from train (3-8)
#returns prediction based on 2nd parameter of class(1) or prob(0)
nb_classifier <- function(column_idx, prediction){
  
  nb_classifier <- textmodel_nb(train_dfm, train[,column_idx])
  
  if(prediction == 1){
    pred <- predict(nb_classifier, 
                    newdata = matched_dfm,
                    type = "class")
  }
  else{
    pred <- predict(nb_classifier, 
                    newdata = matched_dfm,
                    type = "probability")
    
    pred <- pred[,2] 
    
  }
  return(pred)
}

#get predictions of probs and classes
toxic <- nb_classifier(3,0)
toxic_c <- as.numeric(as.character(nb_classifier(3,1)))
severe_toxic <- nb_classifier(4,0)
severe_toxic_c <- as.numeric(as.character(nb_classifier(4,1)))
obscene <- nb_classifier(5,0)
obscene_c <- as.numeric(as.character(nb_classifier(5,1)))
threat <- nb_classifier(6,0)
threat_c <- as.numeric(as.character(nb_classifier(6,1)))
insult <- nb_classifier(7,0)
insult_c <- as.numeric(as.character(nb_classifier(7,1)))
identity_hate <- nb_classifier(8,0)
identity_hate_c <- as.numeric(as.character(nb_classifier(8,1)))

predictions_p <- data.frame(cbind(toxic, severe_toxic, obscene, threat, insult, identity_hate))
head(predictions_p)

predictions_c <- data.frame(cbind(toxic_c, severe_c, obscene_c, threat_c, insult_c, hate_c))
head(predictions_c)

#kaggle output
kaggle <- predictions_p
kaggle$id <- row.names(kaggle)
kaggle <- kaggle[,c(7, 1:6)]
write.csv(kaggle, file = "./out/nb_20201009.csv", row.names = FALSE)

save.image("toxic_comments.RData")


# see how our class predictions perform against test set

#out <- test
#out$toxic_pred <- toxic_c
#out <- out %>%
#  filter(toxic != -1)

#toxic_tab <- table(out$toxic, out$toxic_pred)
#toxic_tab

#confusionMatrix(toxic_tab, mode = "everything")


# generate csv to submit to kaggle


# ----------------------------------------------------------------------------
# train GLMNET Lasso classifiers
# ----------------------------------------------------------------------------

# parallel compute start
#cl <- makeCluster(detectCores()-2, type='PSOCK')
#registerDoParallel(cl)

# fit lasso toxic model
#(start <- Sys.time())
#lasso <- cv.glmnet(x = train_dfm,
 #                  y = as.integer(train$toxic == 1),
  #                 alpha = 1, #lasso
   #                nfold = 5, #5 fold
    #               family = "binomial")
#print(Sys.time() - start)

#https://tutorials.quanteda.io/machine-learning/regression/










