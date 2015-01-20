###########################
# Author:
# Daniel P. Martin
# 
# Title:
# Recursive partitioning 
# example on College data
# for diss proposal
###########################

rm(list = ls())

# setwd to Proposal folder

library(ISLR)
library(rpart)
library(randomForest)
library(party)
library(partykit)
library(dplyr)
library(ggplot2)
library(reshape)
library(gridExtra)
library(gridBase)

# set seed for reproducibility
set.seed(42)

data(College)

head(College)
str(College)
summary(College)

# Split into training and test datasets: 2/3 and 1/3
# Note this is just for the purposes of the simple example,
# when actually splitting it is best to do cross-validation on a
# secondary step before examining test performance

trainIDs <- sample(1:777, 777*(2/3), replace = FALSE)

train <- College[trainIDs, ]
test <- College[-trainIDs, ]

cart <- rpart(Grad.Rate ~ ., data = train)

# Extract cp value that corresponds to the 1SE rule
# First, find minimum xerror value

cptable <- as.data.frame(cart$cptable)

# Get error and SE

min_err <- cptable[which.min(cptable$xerror), c("xerror", "xstd")]

# Extract smallest row number still within 1SE. If none are, it will extract itself
within_rule <- which(cptable$xerror < sum(min_err))[1]

# Get new cp and prune the tree

# cart_prune <- prune(cart, cp = cptable[within_rule, "CP"])
cart_prune <- prune(cart, cp = cptable[which.min(cptable$xerror), "CP"])

pdf("Figures/Chapter02/cart_tree.pdf", width = 10, height = 10)
plot(as.party(cart_prune))
dev.off()

##########################
# Figure 1:
# Sample split with tree

# Plot a slightly larger tree for first example
# to highlight potential overfitting

example_tree <- as.party(prune(cart, cp = .024))

# Plot exact subsection partition using ggplot

part_plot <- ggplot(aes(x = Outstate, y = Top10perc), data = train) + 
  geom_point(aes(size = Grad.Rate), color = "#595959") +
  geom_vline(xintercept = c(7392, 10218.5, 16791), size = 1.5) + 
  geom_segment(aes(x = 10218.5, y = 16.5, xend = 16791, yend = 16.5), size = 1.5) +
  labs(x = "\nOut of State Tuition", y = "Students in Top 10% of High School\n") +
  theme_bw() + guides(size = guide_legend(title = "Grad\nRate")) + 
  annotate("text", label = "Node 3", x = 3500, y = 75, size = 6, fontface = "bold") +
  annotate("text", label = "Node 4", x = 8750, y = 90, size = 6, fontface = "bold") +
  annotate("text", label = "Node 7", x = 15000, y = 0, size = 6, fontface = "bold") +
  annotate("text", label = "Node 8", x = 14500, y = 90, size = 6, fontface = "bold") +
  annotate("text", label = "Node 9", x = 20000, y = 15, size = 6, fontface = "bold") +
  theme(axis.title = element_text(size = 22),
        axis.text = element_text(size = 18),
        legend.title = element_text(size = 18),
        legend.text = element_text(size = 16))

pdf("Figures/Chapter02/part_plot.pdf", height = 10, width = 10)
part_plot
dev.off()

pdf("Figures/Chapter02/example_tree.pdf", height = 10, width = 10)
plot(example_tree)
dev.off()

################
# Figure 3:
# 1SE Rule and 
# error rates
################

# Perform sim for MSE for training, testing, and CV to highlight the bias-variance tradeoff

sim_train_results <- list()
sim_test_results <- list()

for(num_sim in 1:100){
  
  sim_train_MSE <- c()
  sim_test_MSE <- c()
  
  sim_train_ids <- sample(1:777, 777*(2/3), replace = FALSE)
  sim_train <- College[sim_train_ids, ]
  sim_test <- College[-sim_train_ids, ]
  
  for(num_split in 1:8){
    
    sim_cart <- rpart(Grad.Rate ~ ., data = sim_train)
    
    if(num_split %in% sim_cart$cptable[, "nsplit"]){
      
      sim_prune <- prune(sim_cart, sim_cart$cptable[which(num_split == sim_cart$cptable[, "nsplit"]), "CP"])
      sim_train_MSE[num_split] <- mean((sim_train$Grad.Rate - predict(sim_prune, sim_train))^2)
      sim_test_MSE[num_split] <- mean((sim_test$Grad.Rate - predict(sim_prune, sim_test))^2)

    } else{
      
      sim_train_MSE[num_split] <- NA
      sim_test_MSE[num_split] <- NA
       
    }
    
  }
  
  sim_train_results[[num_sim]] <- sim_train_MSE
  sim_test_results[[num_sim]] <- sim_test_MSE
  
  print(num_sim)
  
}

total_train <- do.call(rbind, sim_train_results)
total_test <- do.call(rbind, sim_test_results)

bias_var_data <- data.frame(treeDepth = rep(0:7, 2),
                            Method = rep(c("Training", "Test"), each = 8),
                            MSE = c(apply(total_train, 2, mean, na.rm = TRUE),
                                          apply(total_test, 2, mean, na.rm = TRUE)),
                            SD = c(apply(total_train, 2, sd, na.rm = TRUE),
                                   apply(total_test, 2, sd, na.rm = TRUE)))

bias_var_data$Method <- factor(bias_var_data$Method, levels = c("Training", "Test"))

bias_var_plot <- ggplot(aes(x = treeDepth, y = MSE, group = Method), data = bias_var_data) + 
  geom_point(aes(shape = Method), position = position_dodge(width = 0.25), size = 3) + 
  geom_line(aes(linetype = Method), position = position_dodge(width = 0.25)) +
  geom_errorbar(aes(ymax = MSE + SD, ymin = MSE - SD), position = position_dodge(width = 0.25), width = 0.3) +
  scale_x_continuous(breaks = 0:7) + 
  theme_bw() + labs(x = "\nNumber of Splits\n", y = "Mean-Squared Error\n") + 
  theme(axis.title = element_text(size = 22),
        axis.text = element_text(size = 18),
        legend.title = element_text(size = 18),
        legend.text = element_text(size = 16),
        legend.position = "top",
        legend.key.width = unit(3,"line"))

pdf("Figures/Chapter02/bias_var_plot.pdf", height = 10, width = 10)
bias_var_plot
dev.off()

# Plot 1-SE rule

cp_data <- as.data.frame(cart$cptable)
cp_data <- cp_data[cp_data$nsplit <= 7, ]
labels <- paste(cp_data$nsplit, paste0("(", round(cp_data$CP, 4), ")"), sep = "\n")

cp_plot <- ggplot(aes(x = factor(nsplit), y = xerror, group = 1), data = cp_data) + 
  geom_point(shape = 1, size = 3) + geom_line(size = .5) +
  geom_errorbar(aes(ymax = xerror + xstd, ymin = xerror - xstd, width = 0.2)) + theme_bw() +
  geom_hline(yintercept = sum(cp_data[3, c("xerror", "xstd")]), linetype = "dashed", color = "#595959") +
  labs(x = "\nNumber of Splits\n(Complexity Parameter)", y = "Cross-Validated Error Rate\n") + 
  scale_x_discrete(breaks = cp_data$nsplit, labels = labels) +
  theme(axis.title = element_text(size = 22),
        axis.text = element_text(size = 18),
        legend.title = element_text(size = 18),
        legend.text = element_text(size = 16))

pdf("Figures/Chapter02/cp_plot.pdf", height = 10, width = 10)
cp_plot
dev.off()

##############################
# Comparison of Gini,
# entropy, and classification
##############################

compare_cat <- data.frame(p = seq(0.0, 1.0, 0.005))

compare_cat$misclassification <- apply(data.frame(compare_cat$p, 1 - compare_cat$p), 1, min)
compare_cat$gini <- 2 * compare_cat$p * (1 - compare_cat$p)
compare_cat$entropy <- -1 * compare_cat$p * log2(compare_cat$p) - (1 - compare_cat$p) * log2(1 - compare_cat$p)
compare_cat$entropy <- compare_cat$entropy/2
compare_cat$entropy[c(1, nrow(compare_cat))] <- 0

compare_cat_long <- melt(compare_cat, id.vars = "p")

compare_cat_plot <- ggplot(aes(x = p, y = value), data = compare_cat_long) + 
  geom_line(aes(linetype = variable), size = 1.5) + theme_bw() + 
  scale_linetype_manual(values = c("solid", "longdash", "dotted")) +
  labs(x = "\np", y = "Value\n") + 
  theme(axis.title = element_text(size = 22),
        axis.text = element_text(size = 18),
        legend.title = element_blank(),
        legend.text = element_text(size = 16),
        legend.position = "top",
        legend.key.width = unit(3,"line"))

pdf("Figures/Chapter02/compare_cat_long.pdf", height = 8, width = 10)
compare_cat_plot
dev.off()

######################
# Condition inference
# trees
######################

# Only ctree plot is needed

citree <- ctree(Grad.Rate ~ ., data = train)

pdf("Figures/Chapter02/ci_tree.pdf", width = 16, height = 7)
plot(citree)
dev.off()

######################
# Random Forest
######################

rf <- randomForest(Grad.Rate ~ ., data = train)

var_imp <- data.frame(Variable = row.names(rf$importance),
                      rf$importance)
var_imp <- arrange(var_imp, desc(IncNodePurity))
var_imp$Variable <- factor(var_imp$Variable, levels = rev(var_imp$Variable))

################
# Plot variable 
# importance
################

var_imp_plot <- ggplot(aes(x = IncNodePurity, y = Variable), data = var_imp) + geom_point(size = 3) + theme_bw() + 
  theme(text = element_text(size = 20)) + labs(x = "\nIncrease in Node Purity", y = "Variable\n") + 
  scale_x_continuous(breaks = seq(0, 30000, 5000)) +
  theme(axis.title = element_text(size = 22),
        axis.text = element_text(size = 18),
        legend.title = element_blank(),
        legend.text = element_text(size = 16))

pdf("Figures/Chapter02/var_imp_plot.pdf", height = 10, width = 10)
var_imp_plot
dev.off()

###############
# Plot partial
# dependence
###############

partialPlot(x = rf, pred.data = train, x.var = "Outstate")

# Create partial dependence plot for Out of State

var1_vals <- seq(from = min(train$Outstate), to = max(train$Outstate), by = (max(train$Outstate) - min(train$Outstate))/19)

var1_rep <- train[rep(1:nrow(train), 20), ]

var1_rep$Outstate <- rep(var1_vals, each = nrow(train))

system.time(var1_pred <- predict(rf, var1_rep))
var1_rep$pred <- var1_pred

var1_agg <- group_by(var1_rep, Outstate) %>%
  summarise(mean_pred = mean(pred))

var1_max <- max(var1_agg$mean_pred)
var1_min <- min(var1_agg$mean_pred)

var1_space <- (var1_max - var1_min)/10

var1_part_dep <- ggplot(aes(x = Outstate, y = mean_pred), data = var1_agg) +
  geom_line(size = 1.5) + theme_bw() + labs(x = "\nOut-of-State Tuition", y = "Mean Prediction\n") +
  geom_rug(aes(x = Outstate, y = Grad.Rate), data = train, sides = "b") +
  coord_cartesian(ylim = c(var1_min - var1_space, var1_max + var1_space)) + 
  theme(axis.title = element_text(size = 22),
        axis.text = element_text(size = 18),
        legend.title = element_blank(),
        legend.text = element_text(size = 16))

pdf("Figures/Chapter02/part_dep_Outstate.pdf", height = 8, width = 10)
var1_part_dep
dev.off()

# Create a 3d plot, then a 2d contour (for Outstate and perc.alumni)

var2_vals <- seq(from = min(train$perc.alumni), to = max(train$perc.alumni), by = (max(train$perc.alumni) - min(train$perc.alumni))/19)

two_vals <- expand.grid(var1_vals, var2_vals)
two_vals <- arrange(two_vals, Var1, Var2)

two_rep <- train[rep(1:nrow(train), nrow(two_vals)), ]

two_rep$Outstate <- rep(two_vals$Var1, each = nrow(train))
two_rep$perc.alumni <- rep(two_vals$Var2, each = nrow(train))

system.time(two_pred <- predict(rf, two_rep))
two_rep$pred <- two_pred

two_agg <- group_by(two_rep, Outstate, perc.alumni) %>%
  summarise(mean_pred = mean(pred))

z <- matrix(two_agg$mean_pred, nrow = length(var1_vals), byrow = TRUE)

# Set color

jet.colors <- colorRampPalette( c("#ffffff", "#2a2a2a") ) 

# Generate the desired number of colors from this palette
nbcol <- 100
color <- jet.colors(nbcol)

# Compute the z-value at the facet centres
zfacet <- z[-1, -1] + 
  z[-1, -1 * length(var1_vals)] + 
  z[-1 * length(var2_vals), -1] + 
  z[-1 * length(var1_vals), -1 * length(var2_vals)]

# Recode facet z-values into color indices
facetcol <- cut(zfacet, nbcol)

pdf("Figures/Chapter02/plot_3d.pdf", height = 10, width = 10)
persp(x = var1_vals, y = var2_vals, z = z, theta = -45,
      xlab = "\nOut of State Tuition", ylab = "\nPercentage Alumni Donating", zlab = "\nPredicted Value",
      cex.lab = 1.5,
      ticktype = "detailed",
      col = color[facetcol])
dev.off()

# Contour plot (for later)

# contour <- ggplot(aes(x = Outstate, y = perc.alumni, z = mean_pred), data = two_agg) + geom_tile(aes(fill = mean_pred)) + 
#   scale_fill_gradient(name = "Predicted\nValue", low = "#ffffff", high = "#2a2a2a") +
#   labs(x = "\nOut of State Tuition", y = "perc.alumni\n") +
#   stat_contour() + theme_bw() +
#   theme(axis.title = element_text(size = 22),
#       axis.text = element_text(size = 18),
#       legend.title = element_text(size = 18),
#       legend.text = element_text(size = 16))
# 
# pdf("Figures/Chapter02/contour_plot.pdf", height = 10, width = 10)
# contour
# dev.off()

################
# Plot test set
# approximation
################

part_dep_high <- lapply(as.character(var_imp$Variable[1:]), function(x){
  
  data.frame(var = x,
             type = c(rep('Actual', nrow(train)), rep('Predicted', nrow(train))),
             value = c(test[, x], test[, x]),
             Grad.Rate = c(test$Grad.Rate, as.vector(predict(rf, test))))
  
})

part_data_high <- data.frame(var = "Outstate",
                             type = c(rep('Actual', nrow(train)), rep('Predicted', nrow(train))),
                             value = c(test[, "Outstate"], test[, "Outstate"]),
                             Grad.Rate = c(test$Grad.Rate, as.vector(predict(rf, test))))

testcheck_plot <- ggplot(data = part_data_high, aes(x = value, y = Grad.Rate)) + 
  geom_smooth(color = "black", size = 1.5) + facet_grid(. ~ type) + theme_bw() + 
  labs(x = "\nOut-of-State Tuition", y = "Graduation Rate\n") + 
  theme(axis.title = element_text(size = 26),
        axis.text = element_text(size = 22),
        strip.text = element_text(size = 22),
        legend.title = element_text(size = 22),
        legend.text = element_text(size = 20))

pdf("Figures/Chapter02/testcheck_plot.pdf", height = 10, width = 20)
testcheck_plot
dev.off()

########################
# CFOREST variable
# importance
########################

ciforest <- cforest(Grad.Rate ~ ., data = train)

sort(varimp(ciforest))

########################
# Comparison among the 
# four methods
########################

# Repeatedly split into training and test, record the performance of all four methods

cart_test <- c()
ctree_test <- c()
rf_test <- c()
cforest_test <- c()

system.time(for(num_sim in 1:100){
  
  sim_train_ids <- sample(1:777, 777*(2/3), replace = FALSE)
  sim_train <- College[sim_train_ids, ]
  sim_test <- College[-sim_train_ids, ]
  
  sim_cart_mod <- rpart(Grad.Rate ~ ., data = sim_train)
  
  # Prune CART
  sim_cptable <- as.data.frame(sim_cart_mod$cptable)
  sim_cart_prune <- prune(sim_cart_mod, cp = sim_cptable[which.min(sim_cptable$xerror), "CP"])

  sim_ctree_mod <- ctree(Grad.Rate ~ ., data = sim_train)
  sim_rf_mod <- randomForest(Grad.Rate ~ ., data = sim_train)
  sim_cforest_mod <- cforest(Grad.Rate ~ ., data = sim_train)
  
  cart_test[num_sim] <- mean((sim_test$Grad.Rate - predict(sim_cart_prune, sim_test))^2)
  ctree_test[num_sim] <- mean((sim_test$Grad.Rate - predict(sim_ctree_mod, sim_test))^2)
  rf_test[num_sim] <- mean((sim_test$Grad.Rate - predict(sim_rf_mod, sim_test))^2)
  cforest_test[num_sim] <- mean((sim_test$Grad.Rate - predict(sim_cforest_mod, sim_test, OOB = TRUE))^2)
  
  print(num_sim)
  
})

round(mean(cart_test), 2)
round(sd(cart_test), 2)

round(mean(ctree_test), 2)
round(sd(ctree_test), 2)

round(mean(rf_test), 2)
round(sd(rf_test), 2)

round(mean(cforest_test), 2)
round(sd(cforest_test), 2)

