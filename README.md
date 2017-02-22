**Python implementation of the fixed effects estimator**

Description  
* Includes two methods, fit and predict. Computes and stores coefficients of regressors, intercept and unobserved heterogenity intercepts. Other stats like standard errors or p-values are not included  
* Sklearn-based
    * Uses sklearn BaseEstimator and RegressorMixin as base classes
    * Compatibility with cross validation classes may not work (not tested)  
* Implements FE à la Stata, basically ensuring that the sum of unobserved heterogeneity intercets sum to 0, which eases interpretation. See details [here](www.stata.com/support/faqs/statistics/intercept-in-fixed-effects-model)