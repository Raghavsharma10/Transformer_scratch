def get_gain(data, attr, class_attr,
    method=DEFAULT_DISCRETE_METRIC,
    only_sub=0, prefer_fewer_values=False, entropy_func=None):
    """
    Calculates the information gain (reduction in entropy) that would
    result by splitting the data on the chosen attribute (attr).
    
    Parameters:
    
    prefer_fewer_values := Weights the gain by the count of the attribute's
        unique values. If multiple attributes have the same gain, but one has
        slightly fewer attributes, this will cause the one with fewer
        attributes to be preferred.
    """
    entropy_func = entropy_func or entropy
    val_freq = defaultdict(float)
    subset_entropy = 0.0

    # Calculate the frequency of each of the values in the target attribute
    for record in data:
        val_freq[record.get(attr)] += 1.0

    # Calculate the sum of the entropy for each subset of records weighted
    # by their probability of occuring in the training set.
    for val in val_freq.keys():
        val_prob = val_freq[val] / sum(val_freq.values())
        data_subset = [record for record in data if record.get(attr) == val]
        e = entropy_func(data_subset, class_attr, method=method)
        subset_entropy += val_prob * e
        
    if only_sub:
        return subset_entropy

    # Subtract the entropy of the chosen attribute from the entropy of the
    # whole data set with respect to the target attribute (and return it)
    main_entropy = entropy_func(data, class_attr, method=method)
    
    # Prefer gains on attributes with fewer values.
    if prefer_fewer_values:
#        n = len(val_freq)
#        w = (n+1)/float(n)/2
        #return (main_entropy - subset_entropy)*w
        return ((main_entropy - subset_entropy), 1./len(val_freq))
    else:
        return (main_entropy - subset_entropy)