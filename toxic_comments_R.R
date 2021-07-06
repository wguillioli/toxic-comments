# toxic comments

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


# ----------------------------------------------------------------------------
# create corpus and document feature matrix
# ----------------------------------------------------------------------------

# corpus with all comments (train + test)
comments_corpus <- corpus(comments_df,
                          text_field = "comment_text",
                          docid_field	 = "id")

# basic exploration
summary(comments_corpus,10)
docvars(comments_corpus)[1:10,]
docnames(comments_corpus)[1:25]
ndoc(comments_corpus)

# just in case, have a corpus of train and test separately
#train_corpus <- corpus_subset(comments_corpus, group == "train")
#test_corpus <- corpus_subset(comments_corpus, group == "test")

# create dfm for all comments
comments_dfm <- dfm(comments_corpus,
                    remove_punct = TRUE,
                    remove_symbols = TRUE,
                    remove_numbers = TRUE,
                    remove_separators = TRUE,
                    tolower = TRUE,
                    remove = stopwords("english"), 
                    stem = TRUE, 
                    verbose = TRUE)

# keep tokens than appear at least X times in all docs
comments_dfm <- dfm_trim(comments_dfm, min_termfreq = 2)

dim(comments_dfm)
object.size(comments_dfm)

topfeatures(comments_dfm, 50)

comments_dfm[1:10, 1:10]

# remove additional words

#tf-idf?


# ----------------------------------------------------------------------------
# some basic plots from train dfm
# ----------------------------------------------------------------------------

# split dfm train/test
train_dfm <- dfm_subset(comments_dfm, group == "train")
test_dfm <- dfm_subset(comments_dfm, group == "test")

# top 20 tokens of train
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

# word cloud top 100 words by toxic
train_dfm_grouped <- dfm(train_dfm, groups = "toxic")
set.seed(132)
textplot_wordcloud(train_dfm_grouped, comparison = TRUE, max_words = 200)

# get other metrics like sentiment and appt to dfm
#c <- seq(1:nrow(comments_dfm))
#temp <- cbind(c, comments_dfm)
#dim(temp)
#temp[1:10, 1:10]

# ----------------------------------------------------------------------------
# train NB classifiers
# ----------------------------------------------------------------------------

#function that gets column to predict from train
#returns prediction based on 2nd parameter
nb_classifier <- function(column_idx, prediction){
  #column_idx from 3-8 based on train columns
  #prediction = 1 return class, else return probability
  
  nb_classifier<-textmodel_nb(train_dfm, train[,column_idx])
  
  if(prediction == 1){
    pred <- predict(nb_classifier, 
                    newdata = test_dfm,
                    type = "class")
  }
  else{
    pred <- predict(nb_classifier, 
                    newdata = test_dfm,
                    type = "probability")
    
    pred <- pred[,2] 
    
  }
  
  return(pred)
}

#get predictions of probs and classes
toxic_p <- nb_classifier(3,0)
toxic_c <- as.numeric(as.character(nb_classifier(3,1)))
severe_p <- nb_classifier(4,0)
severe_c <- as.numeric(as.character(nb_classifier(4,1)))
obscene_p <- nb_classifier(5,0)
obscene_c <- as.numeric(as.character(nb_classifier(5,1)))
threat_p <- nb_classifier(6,0)
threat_c <- as.numeric(as.character(nb_classifier(6,1)))
insult_p <- nb_classifier(7,0)
insult_c <- as.numeric(as.character(nb_classifier(7,1)))
hate_p <- nb_classifier(8,0)
hate_c <- as.numeric(as.character(nb_classifier(8,1)))

predictions_p <- data.frame(cbind(toxic_p, severe_p, obscene_p, threat_p, insult_p, hate_p))
head(predictions_p)

predictions_c <- data.frame(cbind(toxic_c, severe_c, obscene_c, threat_c, insult_c, hate_c))
head(predictions_c)

# see how our class predictions perform against test set

out <- test
out$toxic_pred <- toxic_c
out <- out %>%
  filter(toxic != -1)

toxic_tab <- table(out$toxic, out$toxic_pred)
toxic_tab

confusionMatrix(toxic_tab, mode = "everything")


# generate csv to submit to kaggle


# ----------------------------------------------------------------------------
# train GLMNET Lasso classifiers
# ----------------------------------------------------------------------------

# parallel compute start
cl <- makeCluster(detectCores()-2, type='PSOCK')
registerDoParallel(cl)

# fit lasso toxic model
(start <- Sys.time())
lasso <- cv.glmnet(x = train_dfm,
                   y = as.integer(train$toxic == 1),
                   alpha = 1, #lasso
                   nfold = 5, #5 fold
                   family = "binomial")
print(Sys.time() - start)










