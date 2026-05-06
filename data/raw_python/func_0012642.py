def subject_predictions(fm, field='SUBJECTINDEX',
                        method=predict_fixation_duration, data=None):
    '''
    Calculates the saccadic momentum effect for individual subjects.

    Removes any effect of amplitude differences.

    The parameters are fitted on unbinned data. The effects are
    computed on binned data. See e_dist and e_angle for the binning
    parameter.
    '''
    if data is None:
        fma, dura, faa, adsa, ldsa = prepare_data(fm, dur_cap=700, max_back=5)
        adsa = adsa[0]
        ldsa = ldsa[0]
    else:
        fma, dura, faa, adsa, ldsa = data
    fma = fma.copy()  # [ones(fm.x.shape)]
    sub_effects = []
    sub_predictions = []
    parameters = []
    for i, fmsub in enumerate(np.unique(fma.field(field))):
        id = fma.field(field) == fmsub
        #_, dur, fa, ads, lds = prepare_data(fmsub, dur_cap = 700, max_back=5)
        dur, fa, ads, lds = dura[id], faa[id], adsa[id], ldsa[id]
        params = {}
        _ = method(dur, fa, lds, params=params)
        ps = params['v0']
        ld_corrected = leastsq_only_dist(lds, ps[4], ps[5])
        prediction = leastsq_only_angle(fa, ps[0], ps[1], ps[2], ps[3])
        sub_predictions += [saccadic_momentum_effect(prediction, fa)]
        sub_effects += [saccadic_momentum_effect(dur - ld_corrected, fa)]
        parameters += [ps]
    return np.array(sub_effects), np.array(sub_predictions), parameters