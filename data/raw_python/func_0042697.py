def F_oneway(*lists):
    """
Performs a 1-way ANOVA, returning an F-value and probability given
any number of groups.  From Heiman, pp.394-7.

Usage:   F_oneway(*lists)    where *lists is any number of lists, one per
                                  treatment group
Returns: F value, one-tailed p-value
"""
    a = len(lists)           # ANOVA on 'a' groups, each in it's own list
    means = [0] * a
    vars = [0] * a
    ns = [0] * a
    alldata = []
    tmp = lists
    means = map(mean, tmp)
    vars = map(var, tmp)
    ns = map(len, lists)
    for i in range(len(lists)):
        alldata = alldata + lists[i]
    bign = len(alldata)
    sstot = ss(alldata) - (square_of_sums(alldata) / float(bign))
    ssbn = 0
    for list in lists:
        ssbn = ssbn + square_of_sums(list) / float(len(list))
    ssbn = ssbn - (square_of_sums(alldata) / float(bign))
    sswn = sstot - ssbn
    dfbn = a - 1
    dfwn = bign - a
    msb = ssbn / float(dfbn)
    msw = sswn / float(dfwn)
    f = msb / msw
    prob = fprob(dfbn, dfwn, f)
    return f, prob