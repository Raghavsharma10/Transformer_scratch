def chao_shen(q):
    """
    Computes some terms needed for the Chao-Shen KL correction.
    """
    yx = q[q > 0] # remove bins with zero counts
    n = np.sum(yx)
    p = yx.astype(float)/n
    f1 = np.sum(yx == 1) # number of singletons in the sample
    if f1 == n: # avoid C == 0
        f1 -= 1
    C = 1 - (f1/n) # estimated coverage of the sample
    pa = C * p  # coverage adjusted empirical frequencies
    la = (1 - (1 - pa) ** n)  # probability to see a bin (species) in the sample
    H = -np.sum((pa * np.log2(pa)) / la)
    return (H, pa, la)