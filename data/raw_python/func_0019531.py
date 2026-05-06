def seqcor(m1, m2, seq=None):
    """Calculates motif similarity based on Pearson correlation of scores.

    Based on Kielbasa (2015) and Grau (2015).
    Scores are calculated based on scanning a de Bruijn sequence of 7-mers.
    This sequence is taken from ShortCAKE (Orenstein & Shamir, 2015). 
    Optionally another sequence can be given as an argument.

    Parameters
    ----------
    m1 : Motif instance
        Motif 1 to compare.
    
    m2 : Motif instance
        Motif 2 to compare.
    
    seq : str, optional
        Sequence to use for scanning instead of k=7 de Bruijn sequence.
    
    Returns
    -------
    score, position, strand
    """
    l1 = len(m1)
    l2 = len(m2)

    l = max(l1, l2)

    if seq is None:
        seq = RCDB 
    
    L = len(seq)

    # Scan RC de Bruijn sequence
    result1 = pfmscan(seq, m1.pwm, m1.pwm_min_score(), len(seq), False, True)
    result2 = pfmscan(seq, m2.pwm, m2.pwm_min_score(), len(seq), False, True)
    
    # Reverse complement of motif 2
    result3 = pfmscan(seq, m2.rc().pwm, m2.rc().pwm_min_score(), len(seq), False, True)
    
    result1 = np.array(result1)
    result2 = np.array(result2)
    result3 = np.array(result3)

    # Return maximum correlation
    c = []
    for i in range(l1 - l1 // 3):
        c.append([1 - distance.correlation(result1[:L-l-i],result2[i:L-l]), i, 1])
        c.append([1 - distance.correlation(result1[:L-l-i],result3[i:L-l]), i, -1])
    for i in range(l2 - l2 // 3):
        c.append([1 - distance.correlation(result1[i:L-l],result2[:L-l-i]), -i, 1])
        c.append([1 - distance.correlation(result1[i:L-l],result3[:L-l-i]), -i, -1])
    
    return sorted(c, key=lambda x: x[0])[-1]