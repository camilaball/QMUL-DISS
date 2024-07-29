from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor






mdl7 = GradientBoostingRegressor(n_estimators = 100)
rfecv = RFECV(estimator=mdl7, step=1, cv=5,scoring='neg_mean_squared_error')  
rfecv = rfecv.fit(X_train, y_train)

