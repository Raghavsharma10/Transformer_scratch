def intersubject_scores_random_subjects(fm, category, filenumber, n_train,
                                        n_predict, controls=True,
                                        scale_factor = 1):
    """
    Calculates how well the fixations of n random subjects on one image can
    be predicted with the fixations of m other random subjects.

    Notes
        Function that uses intersubject_auc for computing auc.

    Parameters
        fm : fixmat instance
        category : int
            Category from which the fixations are taken.
        filnumber : int
            Image from which fixations are taken.
        n_train : int
            The number of subjects which are used for prediction.
        n_predict : int
            The number of subjects to predict
        controls : bool, optional
            If True (default), n_predict subjects are chosen from the fixmat.
            If False, 1000 fixations are randomly generated and used for
            testing.
        scale_factor : int, optional
            specifies the scaling of the fdm. Default is 1.

    Returns
        tuple : prediction scores
    """
    subjects = np.unique(fm.SUBJECTINDEX)
    if len(subjects) < n_train + n_predict:
        raise ValueError("""Not enough subjects in fixmat""")
    # draw a random sample of subjects for testing and evaluation, according
    # to the specified set sizes (n_train, n_predict)
    np.random.shuffle(subjects)
    predicted_subjects  = subjects[0 : n_predict]
    predicting_subjects = subjects[n_predict : n_predict + n_train]
    assert len(predicting_subjects) == n_train
    assert len(predicted_subjects) == n_predict
    assert [x not in predicting_subjects for x in predicted_subjects]
    return intersubject_scores(fm, category, [filenumber], predicting_subjects,
        [filenumber], predicted_subjects,
        controls, scale_factor)