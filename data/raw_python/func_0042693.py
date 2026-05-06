def friedmanchisquare(*args):
    """
Friedman Chi-Square is a non-parametric, one-way within-subjects
ANOVA.  This function calculates the Friedman Chi-square test for repeated
measures and returns the result, along with the associated probability
value.  It assumes 3 or more repeated measures.  Only 3 levels requires a
minimum of 10 subjects in the study.  Four levels requires 5 subjects per
level(??).

Usage:   lfriedmanchisquare(*args)
Returns: chi-square statistic, associated p-value
"""
    k = len(args)
    if k < 3:
        raise ValueError('Less than 3 levels.  Friedman test not appropriate.')
    n = len(args[0])
    data = map(zip, tuple(args))
    for i in range(len(data)):
        data[i] = rankdata(data[i])
    ssbn = 0
    for i in range(k):
        ssbn = ssbn + sum(args[i]) ** 2
    chisq = 12.0 / (k * n * (k + 1)) * ssbn - 3 * n * (k + 1)
    return chisq, chisqprob(chisq, k - 1)