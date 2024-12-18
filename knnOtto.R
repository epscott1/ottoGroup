library(tidymodels)
library(vroom)
library(caret)
library(themis)

train <- vroom("train.csv")
test <- vroom("test.csv")

my_recipe <- recipe(target ~ ., data = train) %>% 
  step_mutate_at(all_outcomes(), fn = factor, skip = TRUE) %>%
  update_role(id, new_role = "Id") 


prep <- prep(my_recipe)
baked <- bake(prep, new_data = train)

model <- nearest_neighbor(
  neighbors = tune(),
  weight_func = "rectangular",
  dist_power = 2
) %>%
  set_engine("kknn") %>%
  set_mode("classification")

wf <- workflow() %>%
  add_model(model) %>%
  add_recipe(my_recipe)

cv_folds <- vfold_cv(train, v = 5)

knn_grid <- grid_regular(
  neighbors(range = c(50, 75)),  # Number of neighbors to consider
  levels = 3
)

knn_results <- tune_grid(
  wf,
  resamples = cv_folds,
  grid = knn_grid,
  metrics = metric_set(mn_log_loss)
)

=best_knn <- select_best(knn_results, metric = "mn_log_loss")
print(best_knn)

final_knn_workflow <- finalize_workflow(wf, best_knn)

final_knn_fit <- fit(final_knn_workflow, data = train)

predictions <- predict(final_knn_fit, new_data = test, type = "prob")

colnames(predictions) <- gsub(".pred_", "", colnames(predictions))
submission <- predictions %>%
  bind_cols(test %>% select(id)) %>%
  select(id, everything())

vroom_write(submission, "ottoProbs.csv", delim = ",")
