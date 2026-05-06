def labels_at(y_true, y_score, proportion, normalize=False):
    '''
        Return the number of labels encountered in the top  X proportion
    '''
    # Get indexes of scores sorted in descending order
    indexes = np.argsort(y_score)[::-1]

    # Sort true values in the same order
    y_true_sorted = y_true[indexes]

    # Grab top x proportion of true values
    cutoff_index = max(int(len(y_true_sorted) * proportion) - 1, 0)
    # add one to index to grab values including that index
    y_true_top = y_true_sorted[:cutoff_index+1]

    # Count the number of non-nas in the top x proportion
    # we are returning a count so it should be an int
    values = int((~np.isnan(y_true_top)).sum())

    if normalize:
        values = float(values)/(~np.isnan(y_true)).sum()

    return values