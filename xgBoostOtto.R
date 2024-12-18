library(tidymodels)
library(ranger)
library(vroom)
library(xgboost)
library(caret)
library(themis)

#read in data

train <- vroom("train.csv")
test <- vroom("test.csv")

head(train)

#make a recipe

my_recipe <- recipe(target ~., data = train) %>%
  step_mutate_at(all_outcomes(), fn = factor, skip = TRUE) %>%
  update_role(id, new_role = "Id") 

  

prep <- prep(my_recipe)
baked <- bake(prep, new_data = train)

model <- boost_tree(tree_depth = tune(),
                    trees = 50,
                    learn_rate = tune()) %>%
  set_engine("xgboost") %>%
  set_mode("classification")

wf <- workflow() %>%
  add_model(model) %>%
  add_recipe(my_recipe)

cv_folds <- vfold_cv(train, v = 5)

# Tuning grid for xgboost model
boost_grid <- grid_regular(
  tree_depth(range = c(4, 12)),
  learn_rate(range = c(0.01, 0.1)),
  levels = 3
)

# Tuning boost model
boost_results <- tune_grid(
  wf,
  resamples = cv_folds,
  grid = boost_grid,
  metrics = metric_set(mn_log_loss)
)
  
best_boost <- select_best(boost_results, metric = "mn_log_loss")
print(best_boost)

#tree depth of 6, learn rate of 1.26
  
final_boost_workflow <- finalize_workflow(wf, best_boost)
  

final_boost_fit <- fit(final_boost_workflow, data = train)

predictions <- predict(final_boost_fit, new_data = test, type = "prob")

colnames(predictions) <- gsub(".pred_", "", colnames(predictions))
submission <- predictions %>%
  bind_cols(test %>% select(id)) %>%
  select(id, everything())

vroom_write(submission, "ottoProbsxg.csv", delim = ",")
  
#Best score .51 tree depth of 4, learn rate of 1.02
  
  