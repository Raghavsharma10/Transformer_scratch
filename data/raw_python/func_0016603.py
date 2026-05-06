def choose_attribute(data, attributes, class_attr, fitness, method):
    """
    Cycles through all the attributes and returns the attribute with the
    highest information gain (or lowest entropy).
    """
    best = (-1e999999, None)
    for attr in attributes:
        if attr == class_attr:
            continue
        gain = fitness(data, attr, class_attr, method=method)
        best = max(best, (gain, attr))
    return best[1]