def histogram(inlist, numbins=10, defaultreallimits=None, printextras=0):
    """
Returns (i) a list of histogram bin counts, (ii) the smallest value
of the histogram binning, and (iii) the bin width (the last 2 are not
necessarily integers).  Default number of bins is 10.  If no sequence object
is given for defaultreallimits, the routine picks (usually non-pretty) bins
spanning all the numbers in the inlist.

Usage:   lhistogram (inlist, numbins=10, defaultreallimits=None,suppressoutput=0)
Returns: list of bin values, lowerreallimit, binsize, extrapoints
"""
    if (defaultreallimits != None):
        if type(defaultreallimits) not in [list, tuple] or len(defaultreallimits) == 1: # only one limit given, assumed to be lower one & upper is calc'd
            lowerreallimit = defaultreallimits
            upperreallimit = 1.000001 * max(inlist)
        else: # assume both limits given
            lowerreallimit = defaultreallimits[0]
            upperreallimit = defaultreallimits[1]
        binsize = (upperreallimit - lowerreallimit) / float(numbins)
    else:     # no limits given for histogram, both must be calc'd
        estbinwidth = (max(inlist) - min(inlist)) / float(numbins) + 1e-6 # 1=>cover all
        binsize = ((max(inlist) - min(inlist) + estbinwidth)) / float(numbins)
        lowerreallimit = min(inlist) - binsize / 2 # lower real limit,1st bin
    bins = [0] * (numbins)
    extrapoints = 0
    for num in inlist:
        try:
            if (num - lowerreallimit) < 0:
                extrapoints = extrapoints + 1
            else:
                bintoincrement = int((num - lowerreallimit) / float(binsize))
                bins[bintoincrement] = bins[bintoincrement] + 1
        except:
            extrapoints = extrapoints + 1
    if (extrapoints > 0 and printextras == 1):
        print('\nPoints outside given histogram range =', extrapoints)
    return (bins, lowerreallimit, binsize, extrapoints)