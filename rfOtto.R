library(tidymodels)
library(ranger)
library(vroom)
library(caret)
library(themis)

train <- vroom("train.csv")
test <- vroom("test.csv")

head(train)

my_recipe <- recipe(target ~ ., data = train) %>% 
  step_mutate_at(all_outcomes(), fn = factor, skip = TRUE) %>%
  update_role(id, new_role = "Id") %>%
  step_upsample(target)

prep <- prep(my_recipe)
baked <- bake(prep, new_data = train)

model <- rand_forest(
  mtry = 10,
  trees = 500,
  min_n = 5
) %>%
  set_engine("ranger") %>%
  set_mode("classification")

wf <- workflow() %>%
  add_model(model) %>%
  add_recipe(my_recipe)

final_rf_fit <- fit(wf, data = train)

predictions <- predict(final_rf_fit, new_data = test, type = "prob")

colnames(predictions) <- gsub(".pred_", "", colnames(predictions))
submission <- predictions %>%
  bind_cols(test %>% select(id)) %>%
  select(id, everything())

vroom_write(submission, "ottoProbsTreez.csv", delim = ",")
