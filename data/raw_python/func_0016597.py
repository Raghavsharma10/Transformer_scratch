def entropy(data, class_attr=None, method=DEFAULT_DISCRETE_METRIC):
    """
    Calculates the entropy of the attribute attr in given data set data.
    
    Parameters:
    data<dict|list> :=
        if dict, treated as value counts of the given attribute name
        if list, treated as a raw list from which the value counts will be generated
    attr<string> := the name of the class attribute
    """
    assert (class_attr is None and isinstance(data, dict)) \
        or (class_attr is not None and isinstance(data, list))
    if isinstance(data, dict):
        counts = data
    else:
        counts = defaultdict(float) # {attr:count}
        for record in data:
            # Note: A missing attribute is treated like an attribute with a value
            # of None, representing the attribute is "irrelevant".
            counts[record.get(class_attr)] += 1.0
    len_data = float(sum(cnt for _, cnt in iteritems(counts)))
    n = max(2, len(counts))
    total = float(sum(counts.values()))
    assert total, "There must be at least one non-zero count."
    try:
        #return -sum((count/total)*math.log(count/total,n) for count in counts)
        if method == ENTROPY1:
            return -sum((count/len_data)*math.log(count/len_data, n)
                for count in itervalues(counts) if count)
        elif method == ENTROPY2:
            return -sum((count/len_data)*math.log(count/len_data, n)
                for count in itervalues(counts) if count) - ((len(counts)-1)/float(total))
        elif method == ENTROPY3:
            return -sum((count/len_data)*math.log(count/len_data, n)
                for count in itervalues(counts) if count) - 100*((len(counts)-1)/float(total))
        else:
            raise Exception("Unknown entropy method %s." % method)
    except Exception:
        raise