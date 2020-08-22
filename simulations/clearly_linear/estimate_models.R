#estimate_models.R

estimate_mdl_trad <- function(y, x, reg_spec) {
  
  data <- as.data.frame( cbind(y, x) )
  mdl <- lm(data = data, 
            formula = as.formula(reg_spec))
  
  return(mdl)
}