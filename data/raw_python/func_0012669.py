def prediction_scores(prediction, fm, **kw):
    """
    Evaluates a prediction against fixations in a fixmat with different measures.

    The default measures which are used are AUC, NSS and KL-divergence. This
    can be changed by setting the list of measures with set_scores.
    As different measures need potentially different parameters, the kw
    dictionary can be used to pass arguments to measures. Every named
    argument (except fm and prediction) of a measure that is included in
    kw.keys() will be filled with the value stored in kw.
    Example:

    >>> prediction_scores(P, FM, ctr_loc = (y,x))

    In this case the AUC will be computed with control points (y,x), because
    the measure 'roc_model' has 'ctr_loc' as named argument.

    Input:
        prediction  :   2D numpy array
            The prediction that should be evaluated
        fm  :   Fixmat
            The eyetracking data to evaluate against
    Output:
        Tuple of prediction scores. The order of the scores is determined
        by order of measures.scores.
    """
    if prediction == None:
        return [np.NaN for measure in scores]
    results = []
    for measure in scores:
        (args, _, _, _) = inspect.getargspec(measure)
        if len(args)>2:
            # Filter dictionary, such that only the keys that are
            # expected by the measure are in it
            mdict = {}
            [mdict.update({key:value}) for (key, value) in list(kw.items())
                if key in args]
            score = measure(prediction, fm, **mdict)
        else:
            score = measure(prediction, fm)
        results.append(score)
    return results