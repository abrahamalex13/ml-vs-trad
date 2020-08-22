#CONFIG.R

rm(list=ls())
library(tidyverse)
library(caret)
library(elasticnet)
library(parallel)

set.seed(2347)

n <- 1000 #nobs
p <- 50 #number of possible predictors
n_sim <- 1000 #n experimental trials

source("generate_y.R")
source("estimate_models.R")

name_file_out <- "1k obs.rds"




start <- Sys.time()

#workflow: 
  #generate training x
  #generate true model parameters
  #generate outcome series y
  #estimate models
  #evaluate (test) models



#gen training x -----

means_xp <- rep(0, p)
means_xp[1:4] <- c(72, 15, 7, 35)
sigmas_xp <- rep(1, p)
sigmas_xp[1:4] <- c(2.5, 1.5, .75, 2)

x.l <- lapply(1:p, function(x) rnorm(n, means_xp[x], sigmas_xp[x]))
x <- as.matrix( bind_cols(x.l) )
colnames(x) <- paste("x", 1:p, sep = "")
x_std <- apply(x, MARGIN = 2, FUN = function(x) ( x - mean(x) )/ sd(x) )

# ------------------


#true model parameters
betas_xp <- matrix(0, ncol = 1, nrow = p)
betas_xp[1:4, 1] <- c(-5, 10, 15, 10) 


#gen outcome y; estimate models; evaluate

cl <- makeCluster(detectCores() - 2)
clusterExport(cl, c("n_sim", "n", 
                    "x", "x_std", "means_xp", "sigmas_xp", "betas_xp",
                    "generate_y", "estimate_mdl_trad"))
clusterEvalQ(cl, {
  
  library(tidyverse)
  library(caret)
  library(elasticnet)
  
})

clusterSetRNGStream(cl, 12376)

summ.l <- parLapply(cl, 1:n_sim, xp = x, function(x, xp) {
  
    i <- x
    x <- xp
    
    
    #true outcome y
    sigma_e <- 1
    y <- generate_y(x, betas_xp, sigma_e)
    y_mean <- mean(y)
    y_center <- y - y_mean
    
    
    mdl_lm <- estimate_mdl_trad(y, x, reg_spec = "y ~ x1 + x2 + x3 + x4")
    mdl_ml <- caret::train(x = x_std, y = as.numeric(y_center), method = "enet")

    
    #evaluate and test models ---------
    
    mse_train_mdl_lm <- mean(mdl_lm$residuals^2)
    
    mse_train_mdl_ml <- mean( (as.numeric(y) - 
                               (predict(object = mdl_ml, newdata = x_std) + y_mean) )^2 ) 
    
    
    
    x_test.l <- lapply(1:length(means_xp), function(x) rnorm(n, means_xp[x], sigmas_xp[x]))
    x_test <- as.matrix( bind_cols(x_test.l) )
    colnames(x_test) <- paste("x", 1:ncol(x_test), sep = "")
    x_test_std <- apply(x_test, MARGIN = 2, FUN = function(x) ( x - mean(x) )/ sd(x) )
    
    y_test <- generate_y(x_test, betas_xp, sigma_e)
    
    
    mse_test_mdl_lm <- mean( (as.numeric(y_test) - 
                             predict.lm(object = mdl_lm, 
                                        newdata = as.data.frame(x_test[, c("x1", "x2", "x3", "x4")]) ) )^2 )
   
    mse_test_mdl_ml <- mean( (as.numeric(y_test) - 
                               (predict(object = mdl_ml, newdata = x_test_std) + y_mean) )^2 )
    
    #---------------------
    
    
    
    
    summ <- data.frame("mse_train_mdl_lm" = mse_train_mdl_lm, 
                       "mse_train_mdl_ml" = mse_train_mdl_ml, 
                       "mse_test_mdl_lm" = mse_test_mdl_lm,
                       "mse_test_mdl_ml" = mse_test_mdl_ml)
    
    print(paste("complete sim ", i, sep = ""))
    
    return(summ)

})
stopCluster(cl)


summ <- bind_rows(summ.l)
summ[["iter"]] <- 1:n_sim

end <- Sys.time()
print(paste("time elapsed: ", end - start, sep = ""))

saveRDS(summ, file = paste("results/", name_file_out, sep = ""))



# ggplot(summ) + 
#   geom_line(aes(iter, mse_test_mdl_lm, color = "linear regression")) + 
#   geom_line(aes(iter, mse_test_mdl_ml, color = "machine learning linreg"))
  