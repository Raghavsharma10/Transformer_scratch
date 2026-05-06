def precision_at(y_true, y_score, proportion, ignore_nas=False):
    '''
    Calculates precision at a given proportion.
    Only supports binary classification.
    '''
    # Sort scores in descending order
    scores_sorted = np.sort(y_score)[::-1]

    # Based on the proportion, get the index to split the data
    # if value is negative, return 0
    cutoff_index = max(int(len(y_true) * proportion) - 1, 0)
    # Get the cutoff value
    cutoff_value = scores_sorted[cutoff_index]

    # Convert scores to binary, by comparing them with the cutoff value
    scores_binary = np.array([int(y >= cutoff_value) for y in y_score])
    # Calculate precision using sklearn function
    if ignore_nas:
        precision = __precision(y_true, scores_binary)
    else:
        precision = precision_score(y_true, scores_binary)

    return precision, cutoff_value