#generate_y.R

generate_y <- function(x, betas_xp, sigma_e) {
 
   e <- rnorm(n, sd = sigma_e)
   y <- x %*% betas_xp + e
   colnames(y) <- "y"
   
   return(y)
}