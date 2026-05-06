def fill_missing(x):
    """
    Fills in missing lists (assumes end lists are missing)
    """

    # find subject with max number of lists
    maxlen = max([len(xi) for xi in x])

    subs = []

    for sub in x:
        if len(sub)<maxlen:
            for i in range(maxlen-len(sub)):
                sub.append([])
            new_sub = sub
        else:
            new_sub = sub
        subs.append(new_sub)
    return subs