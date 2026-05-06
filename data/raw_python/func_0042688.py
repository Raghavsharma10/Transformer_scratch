def paired(x, y):
    """
Interactively determines the type of data and then runs the
appropriated statistic for paired group data.

Usage:   lpaired(x,y)
Returns: appropriate statistic name, value, and probability
"""
    samples = ''
    while samples not in ['i', 'r', 'I', 'R', 'c', 'C']:
        print('\nIndependent or related samples, or correlation (i,r,c): ',)
        samples = raw_input()

    if samples in ['i', 'I', 'r', 'R']:
        print('\nComparing variances ...',)
        # USE O'BRIEN'S TEST FOR HOMOGENEITY OF VARIANCE, Maxwell & delaney, p.112
        r = obrientransform(x, y)
        f, p = F_oneway(pstat.colex(r, 0), pstat.colex(r, 1))
        if p < 0.05:
            vartype = 'unequal, p=' + str(round(p, 4))
        else:
            vartype = 'equal'
        print(vartype)
        if samples in ['i', 'I']:
            if vartype[0] == 'e':
                t, p = ttest_ind(x, y, 0)
                print('\nIndependent samples t-test:  ', round(t, 4), round(p, 4))
            else:
                if len(x) > 20 or len(y) > 20:
                    z, p = ranksums(x, y)
                    print('\nRank Sums test (NONparametric, n>20):  ', round(z, 4), round(p, 4))
                else:
                    u, p = mannwhitneyu(x, y)
                    print('\nMann-Whitney U-test (NONparametric, ns<20):  ', round(u, 4), round(p, 4))
        else:  # RELATED SAMPLES
            if vartype[0] == 'e':
                t, p = ttest_rel(x, y, 0)
                print('\nRelated samples t-test:  ', round(t, 4), round(p, 4))
            else:
                t, p = ranksums(x, y)
                print('\nWilcoxon T-test (NONparametric):  ', round(t, 4), round(p, 4))
    else:  # CORRELATION ANALYSIS
        corrtype = ''
        while corrtype not in ['c', 'C', 'r', 'R', 'd', 'D']:
            print('\nIs the data Continuous, Ranked, or Dichotomous (c,r,d): ',)
            corrtype = raw_input()
        if corrtype in ['c', 'C']:
            m, b, r, p, see = linregress(x, y)
            print('\nLinear regression for continuous variables ...')
            lol = [['Slope', 'Intercept', 'r', 'Prob', 'SEestimate'], [round(m, 4), round(b, 4), round(r, 4), round(p, 4), round(see, 4)]]
            pstat.printcc(lol)
        elif corrtype in ['r', 'R']:
            r, p = spearmanr(x, y)
            print('\nCorrelation for ranked variables ...')
            print("Spearman's r: ", round(r, 4), round(p, 4))
        else: # DICHOTOMOUS
            r, p = pointbiserialr(x, y)
            print('\nAssuming x contains a dichotomous variable ...')
            print('Point Biserial r: ', round(r, 4), round(p, 4))
    print('\n\n')
    return None