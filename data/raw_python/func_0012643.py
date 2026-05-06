def intersubject_scores(fm, category, predicting_filenumbers,
                        predicting_subjects, predicted_filenumbers,
                        predicted_subjects, controls = True, scale_factor = 1):
    """
    Calculates how well the fixations from a set of subjects on a set of
    images can be predicted with the fixations from another set of subjects
    on another set of images.

    The prediction is carried out by computing a fixation density map from
    fixations of predicting_subjects subjects on predicting_images images.
    Prediction accuracy is assessed by measures.prediction_scores.

    Parameters
        fm : fixmat instance
        category : int
            Category from which the fixations are taken.
        predicting_filenumbers : list
            List of filenumbers used for prediction, i.e. images where fixations
            for the prediction are taken from.
        predicting_subjects : list
            List of subjects whose fixations on images in predicting_filenumbers
            are used for the prediction.
        predicted_filenumnbers : list
            List of images from which the to be predicted fixations are taken.
        predicted_subjects : list
            List of subjects used for evaluation, i.e subjects whose fixations
            on images in predicted_filenumbers are taken for evaluation.
        controls : bool, optional
            If True (default), n_predict subjects are chosen from the fixmat.
            If False, 1000 fixations are randomly generated and used for
            testing.
        scale_factor : int, optional
            specifies the scaling of the fdm. Default is 1.

    Returns
        auc : area under the roc curve for sets of actuals and controls
        true_pos_rate : ndarray
            Rate of true positives for every given threshold value.
            All values appearing in actuals are taken as thresholds. Uses lower
            sum interpolation.
        false_pos_rate : ndarray
            See true_pos_rate but for false positives.

    """
    predicting_fm = fm[
        (ismember(fm.SUBJECTINDEX, predicting_subjects)) &
        (ismember(fm.filenumber, predicting_filenumbers)) &
        (fm.category == category)]
    predicted_fm = fm[
        (ismember(fm.SUBJECTINDEX,predicted_subjects)) &
        (ismember(fm.filenumber,predicted_filenumbers))&
        (fm.category == category)]
    try:
        predicting_fdm = compute_fdm(predicting_fm, scale_factor = scale_factor)
    except RuntimeError:
        predicting_fdm = None

    if controls == True:
        fm_controls = fm[
            (ismember(fm.SUBJECTINDEX, predicted_subjects)) &
            ((ismember(fm.filenumber, predicted_filenumbers)) != True) &
            (fm.category == category)]
        return measures.prediction_scores(predicting_fdm, predicted_fm,
            controls = (fm_controls.y, fm_controls.x))
    return measures.prediction_scores(predicting_fdm, predicted_fm, controls = None)