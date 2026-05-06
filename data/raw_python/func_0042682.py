def cov(x, y, keepdims=0):
    """
Returns the estimated covariance of the values in the passed
array (i.e., N-1).  Dimension can equal None (ravel array first), an
integer (the dimension over which to operate), or a sequence (operate
over multiple dimensions).  Set keepdims=1 to return an array with the
same number of dimensions as inarray.

Usage:   lcov(x,y,keepdims=0)
"""

    n = len(x)
    xmn = mean(x)
    ymn = mean(y)
    xdeviations = [0] * len(x)
    ydeviations = [0] * len(y)
    for i in range(len(x)):
        xdeviations[i] = x[i] - xmn
        ydeviations[i] = y[i] - ymn
    ss = 0.0
    for i in range(len(xdeviations)):
        ss = ss + xdeviations[i] * ydeviations[i]
    return ss / float(n - 1)