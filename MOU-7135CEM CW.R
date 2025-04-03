# Load required libraries
library(crayon)
library(dplyr)  # For data manipulation (grouping and summarizing)
library(ggplot2) # For visualizations
library(bnlearn) # For Bayesian Network
library(caret)
library(corrplot)
library(Metrics)
library(igraph)

# Heading
Heading <- "Steel Industry Energy Consumption using Bayesian Network in Machine Learning"
styledHeading <- bold(underline(blue(Heading)))
cat(styledHeading, "\n")

# --------------- Importing the dataset ---------------
steeldf <- read.csv("Steel_industry_data.csv")

# --------------- Displaying the number of features ---------------
length(steeldf)

# --------------- Displaying the number of rows for the dataset ---------------
nrow(steeldf)

# --------------- Displaying the summary of the dataframe ---------------
summary(steeldf)
str(steeldf)

# --------------- Checking for missing values ---------------
colSums(is.na(steeldf))

# -------------------------------------------------------------------------

# 1 ------------- Data PreProcessing -------------
preprocessing <- "Data Preprocessing"
styledPreprocessing <- bold(underline(blue(preprocessing)))
cat(styledPreprocessing, "\n")

# 1.1 ---------------- Converting Date to DateTime Format ----------------
steeldf$date <- as.POSIXct(steeldf$date, format = "%d/%m/%y %H:%M")

# 1.2 -----------Scatter Plots----------------
hist(steeldf$Usage_kWh, breaks=50, col="blue", main="Usage_kWh Distribution")
hist(steeldf$Lagging_Current_Reactive.Power_kVarh, breaks = 50, col = "blue", main = "Lagging_Current_Reactive Distribution")
hist(steeldf$Leading_Current_Reactive_Power_kVarh, breaks = 50, col = "blue", main = "Leading_Current_Reactive Distribution" )
hist(steeldf$CO2.tCO2., breaks = 50, col = "blue", main = "CO2 Distribution" )
hist(steeldf$Lagging_Current_Power_Factor, breaks = 50, col = "blue", main = "Lagging_Current_Power_Factor Distribution")
hist(steeldf$Leading_Current_Power_Factor, breaks = 50, col = "blue", main = "Leading_Current_Power_Factor Distribution")
hist(steeldf$NSM, breaks = 50, col = "blue", main = "NSM Distribution")

# 1.3 ----------------Feature(Continous) Vs Target Variable(i.e Usage_kWh)----------------
plot(steeldf$Lagging_Current_Reactive.Power_kVarh, steeldf$Usage_kWh, main="Lagging_Current_Reactive vs Usage_kWh", xlab="Lagging_Current_Reactive", ylab="Usage_kWh", col="red")
plot(steeldf$Leading_Current_Reactive_Power_kVarh, steeldf$Usage_kWh, main="Leading_Current_Reactive vs Usage_kWh", xlab="Leading_Current_Reactive_Power_kVarh", ylab="Usage_kWh", col="red")
plot(steeldf$CO2.tCO2., steeldf$Usage_kWh, main="CO2 vs Usage_kWh", xlab="CO2", ylab="Usage_kWh", col="red")
plot(steeldf$Lagging_Current_Power_Factor, steeldf$Usage_kWh, main="Lagging_Current_Power_Factor vs Usage_kWh", xlab="Lagging_Current_Power_Factor", ylab="Usage_kWh", col="red")
plot(steeldf$Leading_Current_Power_Factor, steeldf$Usage_kWh, main="Leading_Current_Power_Factor vs Usage_kWh", xlab="Leading_Current_Power_Factor", ylab="Usage_kWh", col="red")
plot(steeldf$NSM, steeldf$Usage_kWh, main="NSM vs Usage_kWh", xlab="NSM", ylab="Usage_kWh", col="red")

# 1.4 ---------------- Box Plot ----------------
boxplot(steeldf$Usage_kWh, main="Usage_kWh")
boxplot(steeldf$Lagging_Current_Reactive.Power_kVarh, main="Lagging_Current_Reactive Power")
boxplot(steeldf$CO2.tCO2., main="CO2 Emissions")
boxplot(steeldf$Leading_Current_Reactive_Power_kVarh, main="Leading_Current_Reactive Power")
boxplot(steeldf$Lagging_Current_Power_Factor, main="Lagging_Current_Power_Factor")
boxplot(steeldf$Leading_Current_Power_Factor, main="Leading_Current_Power_Factor")

# 1.5 --------------- Handling Outliers ---------------
removeOutliers <- function(outlier){
  firstQuantile <- quantile(outlier, 0.25)
  thirdQuantile <- quantile(outlier, 0.75)
  
  IQR <- thirdQuantile - firstQuantile
  
  lowerBound <- firstQuantile - 1.5 * IQR
  upperBound <- thirdQuantile + 1.5 * IQR
  
  steeldf_no_outlier <- steeldf[outlier >= lowerBound & outlier <= upperBound, ]
}

steeldf <- removeOutliers(steeldf$Leading_Current_Reactive_Power_kVarh)
nrow(steeldf)

# Handling outliers for all relevant columns
steeldf <- removeOutliers(steeldf$Usage_kWh)
steeldf <- removeOutliers(steeldf$Lagging_Current_Reactive.Power_kVarh)
steeldf <- removeOutliers(steeldf$CO2.tCO2.)
steeldf <- removeOutliers(steeldf$Lagging_Current_Power_Factor)
steeldf <- removeOutliers(steeldf$Leading_Current_Power_Factor)
nrow(steeldf)

# 1.6 ------------- Bar Chart plot for Usage vs LoadType -------------
mean_usage_by_load <- summarize(group_by(steeldf, Load_Type), Mean_Usage = mean(Usage_kWh, na.rm = TRUE))

# Create the bar chart
ggplot(mean_usage_by_load, aes(x = Load_Type, y = Mean_Usage)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  labs(title = "Mean Usage kWh by Load Type",
       x = "Load Type",
       y = "Mean Usage (kWh)")

# 2 ---------------- Feature Engineering -------------------

# 2.1 ------------- Time Of The Day using NSM -------------
steeldf$Time_of_day <- cut(steeldf$NSM,
                           breaks = c(0, 21600, 43200, 64800, 86400),
                           labels = c("MidNight", "Morning", "Afternoon", "Evening"),
                           include.lowest = TRUE)

# 2.2 ------------- Categorizing CO2.tCO2 -------------
steeldf$CO2_Category <- cut(steeldf$CO2.tCO2., breaks = 3, labels = c("Low", "Medium", "High"))

# 2.3 --------------------- Split Date-Time ----------------------------
steeldf$Year <- as.integer(format(steeldf$date, "%Y"))
steeldf$Month <- as.integer(format(steeldf$date, "%m"))
steeldf$Day <- as.integer(format(steeldf$date, "%d"))
steeldf$Hour <- as.integer(format(steeldf$date, "%H"))
steeldf$Minute <- as.integer(format(steeldf$date, "%M"))

steeldf$date <- NULL

# 3 ---------------- Feature Selection -------------------

# 3.1 ------------------ Correlation-Plot --------------------------
# Select only numeric columns
numeric_df <- steeldf[sapply(steeldf, is.numeric)]

# Compute the correlation matrix
correlation_matrix <- cor(numeric_df, use = "complete.obs")



# Plot the correlation matrix
corrplot(correlation_matrix, 
         method = "color", 
         type = "upper", 
         tl.cex = 0.3, 
         tl.col = "black", 
         addCoef.col = "black",
         number.cex = 0.5
        )

# 3.2----------------- Removing unnecessary columns ---------------- 

steeldf <- steeldf %>% select(-Leading_Current_Power_Factor)

# 3.3 ----------------- Remove rows with missing values ----------------- 
steeldf_cleaned <- na.omit(steeldf)
nrow(steeldf_cleaned)
steeldf <- steeldf_cleaned

# 3.4 ----------------- Selecting columns for the Bayesian Network model  ----------------- 
selected_columns <- c("Usage_kWh", "Lagging_Current_Reactive.Power_kVarh", "Leading_Current_Reactive_Power_kVarh", "Lagging_Current_Power_Factor", "WeekStatus", "Day_of_week", "Load_Type", "Time_of_day", "CO2_Category", "Month", "Day", "Hour", "Minute")
regression_df <- steeldf[, selected_columns]

# 4 ---------------- Discretization ----------------
regression_df$Usage_kWh_discrete <- cut(regression_df$Usage_kWh, breaks = 3, labels = c("Low", "Medium", "High"))
regression_df$Lagging_Current_Reactive.Power_kVarh_discrete <- cut(regression_df$Lagging_Current_Reactive.Power_kVarh, breaks = 3, labels = c("Low", "Medium", "High"))
regression_df$Leading_Current_Reactive_Power_kVarh_discrete <- cut(regression_df$Leading_Current_Reactive_Power_kVarh, breaks = 3, labels = c("Low", "Medium", "High"))
regression_df$Lagging_Current_Power_Factor_discrete <- cut(regression_df$Lagging_Current_Power_Factor, breaks = 3, labels = c("Low", "Medium", "High"))

# 4.1 ----------------- Convert factor variables  ----------------- 
factor_cols <- c("WeekStatus", "Day_of_week", "Load_Type", "Time_of_day", "CO2_Category", "Month", "Day", "Hour", "Minute")
regression_df[factor_cols] <- lapply(regression_df[factor_cols], as.factor)


# 5 ---------------- Train-Test-Split ----------------
set.seed(124)
trainRatio <- 0.7
rows <- nrow(regression_df)
trainSize <- floor(trainRatio * rows)
trainIndices <- sample(1:rows, size = trainSize)

trainDF <- regression_df[trainIndices, ]
testDF <- regression_df[-trainIndices, ]

# 6 ---------------- Learn and Fit Bayesian Network ----------------
bn_structure <- hc(trainDF)  # Learn the structure
bn_fitted <- bn.fit(bn_structure, trainDF)  # Fit the parameters

# 7 ---------------- Make Predictions ----------------

set.seed(124)
k <- 10  # Number of folds
folds <- createFolds(regression_df$Usage_kWh, k = k, list = TRUE)

# 7.1 ---------------- HC Algorithm  ---------------- 
# Initialize vectors to store metrics
train_mse_values <- c()
train_rmse_values <- c()
train_r_squared_values <- c()

test_mse_values <- c()
test_rmse_values <- c()
test_r_squared_values <- c()

for (i in 1:k) {
  cat("\n========== Processing Fold:", i, "==========\n")
  
  # Split data into training and test sets
  train_indices <- folds[[i]]
  train_data <- regression_df[train_indices, ]
  test_data <- regression_df[-train_indices, ]
  
  # Learn the structure from training data
  bn_structure <- hc(train_data)
  bn_fitted <- bn.fit(bn_structure, train_data)
  
  # ---------------- TRAINING SET PREDICTIONS ----------------
  train_predictions <- tryCatch({
    predict(bn_fitted, data = train_data, node = "Usage_kWh", method = "bayes-lw")
  }, error = function(e) {
    cat("Error in training prediction for fold", i, ":", e$message, "\n")
    return(NA)
  })
  
  train_actual_values <- train_data$Usage_kWh
  
  # Check if lengths match
  if (length(train_predictions) != length(train_actual_values)) {
    cat("Length mismatch in training data for fold", i, "\n")
    next
  }
  
  # Compute training error metrics
  train_mse <- mean((train_predictions - train_actual_values)^2, na.rm = TRUE)
  train_rmse <- sqrt(train_mse)
  train_rss <- sum((train_predictions - train_actual_values)^2, na.rm = TRUE)
  train_tss <- sum((train_actual_values - mean(train_actual_values, na.rm = TRUE))^2)
  train_r_squared <- 1 - (train_rss / train_tss)
  
  # Store training metrics
  train_mse_values <- c(train_mse_values, train_mse)
  train_rmse_values <- c(train_rmse_values, train_rmse)
  train_r_squared_values <- c(train_r_squared_values, train_r_squared)
  
  cat("Training MSE:", train_mse, "\n")
  cat("Training RMSE:", train_rmse, "\n")
  cat("Training R-squared:", train_r_squared, "\n")
  
  # ---------------- TEST SET PREDICTIONS ----------------
  test_predictions <- tryCatch({
    predict(bn_fitted, data = test_data, node = "Usage_kWh", method = "bayes-lw")
  }, error = function(e) {
    cat("Error in testing prediction for fold", i, ":", e$message, "\n")
    return(NA)
  })
  
  test_actual_values <- test_data$Usage_kWh
  
  # Check if lengths match
  if (length(test_predictions) != length(test_actual_values)) {
    cat("Length mismatch in test data for fold", i, "\n")
    next
  }
  
  # Compute testing error metrics
  test_mse <- mean((test_predictions - test_actual_values)^2, na.rm = TRUE)
  test_rmse <- sqrt(test_mse)
  test_rss <- sum((test_predictions - test_actual_values)^2, na.rm = TRUE)
  test_tss <- sum((test_actual_values - mean(test_actual_values, na.rm = TRUE))^2)
  test_r_squared <- 1 - (test_rss / test_tss)
  
  # Store testing metrics
  test_mse_values <- c(test_mse_values, test_mse)
  test_rmse_values <- c(test_rmse_values, test_rmse)
  test_r_squared_values <- c(test_r_squared_values, test_r_squared)
  
  cat("Testing MSE:", test_mse, "\n")
  cat("Testing RMSE:", test_rmse, "\n")
  cat("Testing R-squared:", test_r_squared, "\n")
}

# ---------------- FINAL AVERAGE METRICS ----------------

cat("\n=================== FINAL AVERAGE METRICS ===================\n")

cat("\n**Average Training Results:**\n")
cat("Average Training MSE:", mean(train_mse_values, na.rm = TRUE), "\n")
cat("Average Training RMSE:", mean(train_rmse_values, na.rm = TRUE), "\n")
cat("Average Training R-squared:", mean(train_r_squared_values, na.rm = TRUE), "\n")

cat("\n**Average Testing Results:**\n")
cat("Average Testing MSE:", mean(test_mse_values, na.rm = TRUE), "\n")
cat("Average Testing RMSE:", mean(test_rmse_values, na.rm = TRUE), "\n")
cat("Average Testing R-squared:", mean(test_r_squared_values, na.rm = TRUE), "\n")

# ---------------- Bayesian Network Structure For HC Algorithm ---------------- 

# Convert the learned network to an igraph object
bn_igraph <- as.igraph(bn_fitted)

# Plot the graph with igraph
plot(bn_igraph, vertex.size=21, vertex.label.cex=0.7, edge.arrow.size=0.5)

graphviz.chart(bn_fitted, type = "barprob",  grid = TRUE,  bar.col = "green",strip.bg = "skyblue")

# 7.2 ------------------------- PC Algorithm -----------------------
# Learn structure with PC algorithm
bn_structure2 <- pc.stable(trainDF)

# Convert PDAG to fully directed DAG
bn_structure2_dag <- cextend(bn_structure2)

# Fit parameters
bn_fitted2 <- bn.fit(bn_structure2_dag, trainDF)

new_predictions <- tryCatch({
  predict(bn_fitted2, data = test_data, node = "Usage_kWh", method = "bayes-lw")
}, error = function(e) {
  cat("Error in prediction for fold", i, ":", e$message, "\n")
  return(NA)
})

# Remove NA values if prediction failed
valid_indices <- !is.na(new_predictions)
actual_values <- test_data$Usage_kWh[valid_indices]
predicted_values <- new_predictions[valid_indices]

# ---------------- Compute error metrics ---------------- 
mae_value  <- mae(actual_values, predicted_values)
mse_value  <- mse(actual_values, predicted_values)
rmse_value <- rmse(actual_values, predicted_values)
r2_value   <- cor(actual_values, predicted_values)^2  # R-squared

# Print results
cat("Mean Absolute Error (MAE):", mae_value, "\n")
cat("Mean Squared Error (MSE):", mse_value, "\n")
cat("Root Mean Squared Error (RMSE):", rmse_value, "\n")
cat("R-squared (R²):", r2_value, "\n")

# ---------------- Bayesian Network Structure For PC Algorithm ---------------- 

bn_igraph2 <- as.igraph(bn_fitted2)

# Plot the graph with igraph
plot(bn_igraph2, vertex.size=21, vertex.label.cex=0.7, edge.arrow.size=0.5)

graphviz.chart(bn_fitted2, type = "barprob",  grid = TRUE,  bar.col = "blue",strip.bg = "orange")






