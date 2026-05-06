def upper_bound(fm, nr_subs = None, scale_factor = 1):
    """
    compute the inter-subject consistency upper bound for a fixmat.

    Input:
        fm : a fixmat instance
        nr_subs : the number of subjects used for the prediction. Defaults
                  to the total number of subjects in the fixmat minus 1
        scale_factor : the scale factor of the FDMs. Default is 1.
    Returns:
        A list of scores; the list contains one dictionary for each measure.
        Each dictionary contains one key for each category and corresponding
        values is an array with scores for each subject.
    """
    nr_subs_total = len(np.unique(fm.SUBJECTINDEX))
    if not nr_subs:
        nr_subs = nr_subs_total - 1
    assert (nr_subs < nr_subs_total)
    # initialize output structure; every measure gets one dict with
    # category numbers as keys and numpy-arrays as values
    intersub_scores = []
    for measure in range(len(measures.scores)):
        res_dict = {}
        result_vectors = [np.empty(nr_subs_total) + np.nan
                            for _ in np.unique(fm.category)]
        res_dict.update(list(zip(np.unique(fm.category), result_vectors)))
        intersub_scores.append(res_dict)
    #compute inter-subject scores for every stimulus, with leave-one-out
    #over subjects
    for fm_cat in fm.by_field('category'):
        cat = fm_cat.category[0]
        for (sub_counter, sub) in enumerate(np.unique(fm_cat.SUBJECTINDEX)):
            image_scores = []
            for fm_single in fm_cat.by_field('filenumber'):
                predicting_subs = (np.setdiff1d(np.unique(
                    fm_single.SUBJECTINDEX),[sub]))
                np.random.shuffle(predicting_subs)
                predicting_subs = predicting_subs[0:nr_subs]
                predicting_fm = fm_single[
                    (ismember(fm_single.SUBJECTINDEX, predicting_subs))]
                predicted_fm = fm_single[fm_single.SUBJECTINDEX == sub]
                try:
                    predicting_fdm = compute_fdm(predicting_fm,
                        scale_factor = scale_factor)
                except RuntimeError:
                    predicting_fdm = None
                image_scores.append(measures.prediction_scores(
                                        predicting_fdm, predicted_fm))
            for (measure, score) in enumerate(nanmean(image_scores, 0)):
                intersub_scores[measure][cat][sub_counter] = score
    return intersub_scores