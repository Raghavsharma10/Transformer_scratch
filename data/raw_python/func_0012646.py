def lower_bound(fm, nr_subs = None, nr_imgs = None, scale_factor = 1):
    """
    Compute the spatial bias lower bound for a fixmat.

    Input:
        fm : a fixmat instance
        nr_subs : the number of subjects used for the prediction. Defaults
                  to the total number of subjects in the fixmat minus 1
        nr_imgs : the number of images used for prediction. If given, the
                  same number will be used for every category. If not given,
                  leave-one-out will be used in all categories.
        scale_factor : the scale factor of the FDMs. Default is 1.
    Returns:
        A list of spatial bias scores; the list contains one dictionary for each
         measure. Each dictionary contains one key for each category and
        corresponding values is an array with scores for each subject.
    """
    nr_subs_total = len(np.unique(fm.SUBJECTINDEX))
    if nr_subs is None:
        nr_subs = nr_subs_total - 1
    assert (nr_subs < nr_subs_total)
    # initialize output structure; every measure gets one dict with
    # category numbers as keys and numpy-arrays as values
    sb_scores = []
    for measure in range(len(measures.scores)):
        res_dict = {}
        result_vectors = [np.empty(nr_subs_total) + np.nan
                            for _ in np.unique(fm.category)]
        res_dict.update(list(zip(np.unique(fm.category),result_vectors)))
        sb_scores.append(res_dict)
    # compute mean spatial bias predictive power for all subjects in all
    # categories
    for fm_cat in fm.by_field('category'):
        cat = fm_cat.category[0]
        nr_imgs_cat = len(np.unique(fm_cat.filenumber))
        if not nr_imgs:
            nr_imgs_current = nr_imgs_cat - 1
        else:
            nr_imgs_current = nr_imgs
        assert(nr_imgs_current < nr_imgs_cat)
        for (sub_counter, sub) in enumerate(np.unique(fm.SUBJECTINDEX)):
            image_scores = []
            for fm_single in fm_cat.by_field('filenumber'):
                # Iterating by field filenumber makes filenumbers
                # in fm_single unique: Just take the first one to get the
                # filenumber for this fixmat
                fn = fm_single.filenumber[0]
                predicting_subs = (np.setdiff1d(np.unique(
                    fm_cat.SUBJECTINDEX), [sub]))
                np.random.shuffle(predicting_subs)
                predicting_subs = predicting_subs[0:nr_subs]
                predicting_fns = (np.setdiff1d(np.unique(
                    fm_cat.filenumber), [fn]))
                np.random.shuffle(predicting_fns)
                predicting_fns = predicting_fns[0:nr_imgs_current]
                predicting_fm = fm_cat[
                    (ismember(fm_cat.SUBJECTINDEX, predicting_subs)) &
                    (ismember(fm_cat.filenumber, predicting_fns))]
                predicted_fm = fm_single[fm_single.SUBJECTINDEX == sub]
                try:
                    predicting_fdm = compute_fdm(predicting_fm,
                        scale_factor = scale_factor)
                except RuntimeError:
                    predicting_fdm = None
                image_scores.append(measures.prediction_scores(predicting_fdm,
                     predicted_fm))
            for (measure, score) in enumerate(nanmean(image_scores, 0)):
                sb_scores[measure][cat][sub_counter] = score
    return sb_scores