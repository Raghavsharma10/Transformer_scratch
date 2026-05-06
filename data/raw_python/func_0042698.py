def F_value(ER, EF, dfnum, dfden):
    """
Returns an F-statistic given the following:
        ER  = error associated with the null hypothesis (the Restricted model)
        EF  = error associated with the alternate hypothesis (the Full model)
        dfR-dfF = degrees of freedom of the numerator
        dfF = degrees of freedom associated with the denominator/Full model

Usage:   lF_value(ER,EF,dfnum,dfden)
"""
    return ((ER - EF) / float(dfnum) / (EF / float(dfden)))