def calc_motif_enrichment(sample, background, mtc=None, len_sample=None, len_back=None):
    """Calculate enrichment based on hypergeometric distribution"""
    
    INF = "Inf"


    if mtc not in [None, "Bonferroni", "Benjamini-Hochberg", "None"]:
        raise RuntimeError("Unknown correction: %s" % mtc)

    sig = {}
    p_value  = {}
    n_sample = {}
    n_back = {}
    
    if not(len_sample):
        len_sample = sample.seqn()
    if not(len_back):
        len_back = background.seqn()

    for motif in sample.motifs.keys():
        p = "NA"
        s = "NA"
        q = len(sample.motifs[motif])
        m = 0
        if(background.motifs.get(motif)):
            m = len(background.motifs[motif])
            n = len_back - m
            k = len_sample
            p = phyper(q - 1, m, n, k) 
            if p != 0:
                s = -(log(p)/log(10))
            else:
                s = INF
        else:
            s = INF
            p = 0.0

        sig[motif] = s
        p_value[motif] = p
        n_sample[motif] = q
        n_back[motif] = m
    
    if mtc == "Bonferroni":
        for motif in p_value.keys():
            if  p_value[motif] != "NA":
                p_value[motif] = p_value[motif] * len(p_value.keys())
                if p_value[motif] > 1:
                    p_value[motif] = 1
    elif mtc == "Benjamini-Hochberg":
        motifs = sorted(p_value.keys(), key=lambda x: -p_value[x])
        l = len(p_value)
        c = l
        for m in motifs:
            if  p_value[m] != "NA":
                p_value[m] = p_value[m] * l / c 
            c -= 1

    return (sig, p_value, n_sample, n_back)