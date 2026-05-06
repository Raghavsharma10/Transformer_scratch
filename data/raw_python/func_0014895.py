def shuffle(seq, random=None):
    r"""Return shuffled *copy* of `seq`."""
    if isinstance(seq, list):
        return ipshuffle(seq[:], random)
    elif isString(seq):
        # seq[0:0] == "" or  u""
        return seq[0:0].join(ipshuffle(list(seq)),random)
    else:
        return type(seq)(ipshuffle(list(seq),random))