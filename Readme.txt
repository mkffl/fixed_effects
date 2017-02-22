Python implementation of the fixed effects estimator

- Using sklearn BaseEstimator and RegressorMixin underlying classes
- Restricted to fit and predict methods. Does not include any statistics
- Implements FE à la Stata, basically ensuring that the sum of unobserved heterogeneity intercets sum to 0, which eases interpretation
-> See http://www.stata.com/support/faqs/statistics/intercept-in-fixed-effects-model/