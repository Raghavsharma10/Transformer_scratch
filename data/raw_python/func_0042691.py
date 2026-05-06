def chisquare(f_obs, f_exp=None):
    """
Calculates a one-way chi square for list of observed frequencies and returns
the result.  If no expected frequencies are given, the total N is assumed to
be equally distributed across all groups.

Usage:   lchisquare(f_obs, f_exp=None)   f_obs = list of observed cell freq.
Returns: chisquare-statistic, associated p-value
"""
    k = len(f_obs)                 # number of groups
    if f_exp == None:
        f_exp = [sum(f_obs) / float(k)] * len(f_obs) # create k bins with = freq.
    chisq = 0
    for i in range(len(f_obs)):
        o = f_obs[i]
        e = f_exp[i]
        chisq = chisq + (o - e) ** 2 / float(e)
    return chisq, chisqprob(chisq, k - 1)