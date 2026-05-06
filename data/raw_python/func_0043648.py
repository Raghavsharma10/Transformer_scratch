def make_predictor(regressor=LassoLarsIC(fit_intercept=False),
                   Selector=GridSearchCV, fourier_degree=(2, 25),
                   selector_processes=1,
                   use_baart=False, scoring='r2', scoring_cv=3,
                   **kwargs):
    """make_predictor(regressor=LassoLarsIC(fit_intercept=False), Selector=GridSearchCV, fourier_degree=(2, 25), selector_processes=1, use_baart=False, scoring='r2', scoring_cv=3, **kwargs)

    Makes a predictor object for use in :func:`get_lightcurve`.

    **Parameters**

    regressor : object with "fit" and "transform" methods, optional
        Regression object used for solving Fourier matrix
        (default ``sklearn.linear_model.LassoLarsIC(fit_intercept=False)``).
    Selector : class with "fit" and "predict" methods, optional
        Model selection class used for finding the best fit
        (default :class:`sklearn.grid_search.GridSearchCV`).
    selector_processes : positive integer, optional
        Number of processes to use for *Selector* (default 1).
    use_baart : boolean, optional
        If True, ignores *Selector* and uses Baart's Criteria to find
        the Fourier degree, within the boundaries (default False).
    fourier_degree : 2-tuple, optional
        Tuple containing lower and upper bounds on Fourier degree, in that
        order (default (2, 25)).
    scoring : str, optional
        Scoring method to use for *Selector*. This parameter can be:
            * "r2", in which case use :math:`R^2` (the default)
            * "mse", in which case use mean square error
    scoring_cv : positive integer, optional
        Number of cross validation folds used in scoring (default 3).

    **Returns**

    out : object with "fit" and "predict" methods
        The created predictor object.
    """
    fourier = Fourier(degree_range=fourier_degree, regressor=regressor) \
              if use_baart else Fourier()
    pipeline = Pipeline([('Fourier', fourier), ('Regressor', regressor)])
    if use_baart:
        return pipeline
    else:
        params = {'Fourier__degree': list(range(fourier_degree[0],
                                                fourier_degree[1]+1))}
        return Selector(pipeline, params, scoring=scoring, cv=scoring_cv,
                        n_jobs=selector_processes)