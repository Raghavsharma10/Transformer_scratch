def get_variance(seq):
    """
    Batch variance calculation.
    """
    m = get_mean(seq)
    return sum((v-m)**2 for v in seq)/float(len(seq))