def fixation_detection(samplemat, saccades, Hz=200, samples2fix=None,
                       respect_trial_borders=False, sample_times=None):
    '''
    Detect Fixation from saccades.

    Fixations are defined as intervals between saccades. This function
    also calcuates start and end times (in ms) for each fixation.
    Input:
        samplemat: datamat
            Contains the recorded samples and associated metadata.
        saccades: ndarray
            Logical vector that is True for samples that belong to a saccade.
        Hz: Float
            Number of samples per second.
        samples2fix: Dict
            There is usually metadata associated with the samples (e.g. the
            trial number). This dictionary can be used to specify how the
            metadata should be collapsed for one fixation. It contains
            field names from samplemat as keys and functions as values that
            return one value when they are called with all samples for one
            fixation. In addition the function can raise an 'InvalidFixation'
            exception to signal that the fixation should be discarded.
    '''
    if samples2fix is None:
        samples2fix = {}
    fixations = ~saccades
    acc = AccumulatorFactory()
    if not respect_trial_borders:
        borders = np.where(np.diff(fixations.astype(int)))[0] + 1
    else:
        borders = np.where(
            ~(np.diff(fixations.astype(int)) == 0) |
            ~(np.diff(samplemat.trial.astype(int)) == 0))[0] + 1

    fixations = 0 * saccades.copy()
    if not saccades[0]:
        borders = np.hstack(([0], borders))
    #lasts,laste = borders[0], borders[1]
    for i, (start, end) in enumerate(zip(borders[0::2], borders[1::2])):

        current = {}
        for k in samplemat.fieldnames():
            if k in list(samples2fix.keys()):
                current[k] = samples2fix[k](samplemat, k, start, end)
            else:
                current[k] = np.mean(samplemat.field(k)[start:end])
        current['start_sample'] = start
        current['end_sample'] = end
        fixations[start:end] = 1
        # Calculate start and end time in ms
        if sample_times is None:
            current['start'] = 1000 * start / Hz
            current['end'] = 1000 * end / Hz
        else:
            current['start'] = sample_times[start]
            current['end'] = sample_times[end]

        #lasts, laste = start,end
        acc.update(current)

    return acc.get_dm(params=samplemat.parameters()), fixations.astype(bool)