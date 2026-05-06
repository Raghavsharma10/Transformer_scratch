def findwithin(data):
    """
Returns an integer representing a binary vector, where 1=within-
subject factor, 0=between.  Input equals the entire data 2D list (i.e.,
column 0=random factor, column -1=measured values (those two are skipped).
Note: input data is in |Stat format ... a list of lists ("2D list") with
one row per measured value, first column=subject identifier, last column=
score, one in-between column per factor (these columns contain level
designations on each factor).  See also stats.anova.__doc__.

Usage:   lfindwithin(data)     data in |Stat format
"""

    numfact = len(data[0]) - 1
    withinvec = 0
    for col in range(1, numfact):
        examplelevel = pstat.unique(pstat.colex(data, col))[0]
        rows = pstat.linexand(data, col, examplelevel)  # get 1 level of this factor
        factsubjs = pstat.unique(pstat.colex(rows, 0))
        allsubjs = pstat.unique(pstat.colex(data, 0))
        if len(factsubjs) == len(allsubjs):  # fewer Ss than scores on this factor?
            withinvec = withinvec + (1 << col)
    return withinvec