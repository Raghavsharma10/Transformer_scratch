def randrange(seq):
    """ Yields random values from @seq until @seq is empty """
    seq = seq.copy()
    choose = rng().choice
    remove = seq.remove
    for x in range(len(seq)):
        y = choose(seq)
        remove(y)
        yield y