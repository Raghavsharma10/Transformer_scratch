def findbeam_semitransparent(data, pri, threshold=0.05):
    """Find beam with 2D weighting of semitransparent beamstop area

    Inputs:
        data: scattering matrix
        pri: list of four: [xmin,xmax,ymin,ymax] for the borders of the beam
            area under the semitransparent beamstop. X corresponds to the column
            index (ie. A[Y,X] is the element of A from the Xth column and the
            Yth row). You can get these by zooming on the figure and retrieving
            the result of axis() (like in Matlab)
        threshold: do not count pixels if their intensity falls below
            max_intensity*threshold. max_intensity is the highest count rate
            in the current row or column, respectively. Set None to disable
            this feature.

    Outputs: bcx,bcy
        the x and y coordinates of the primary beam
    """
    rowmin = np.floor(min(pri[2:]))
    rowmax = np.ceil(max(pri[2:]))
    colmin = np.floor(min(pri[:2]))
    colmax = np.ceil(max(pri[:2]))

    if threshold is not None:
        # beam area on the scattering image
        B = data[rowmin:rowmax, colmin:colmax]
        # print B.shape
        # row and column indices
        Ri = np.arange(rowmin, rowmax)
        Ci = np.arange(colmin, colmax)
        # print len(Ri)
        # print len(Ci)
        Ravg = B.mean(1)  # average over column index, will be a concave curve
        Cavg = B.mean(0)  # average over row index, will be a concave curve
        # find the maxima im both directions and their positions
        maxR = Ravg.max()
        maxRpos = Ravg.argmax()
        maxC = Cavg.max()
        maxCpos = Cavg.argmax()
        # cut off pixels which are smaller than threshold*peak_height
        Rmin = Ri[
            ((Ravg - Ravg[0]) >= ((maxR - Ravg[0]) * threshold)) & (Ri < maxRpos)][0]
        Rmax = Ri[
            ((Ravg - Ravg[-1]) >= ((maxR - Ravg[-1]) * threshold)) & (Ri > maxRpos)][-1]
        Cmin = Ci[
            ((Cavg - Cavg[0]) >= ((maxC - Cavg[0]) * threshold)) & (Ci < maxCpos)][0]
        Cmax = Ci[
            ((Cavg - Cavg[-1]) >= ((maxC - Cavg[-1]) * threshold)) & (Ci > maxCpos)][-1]
    else:
        Rmin = rowmin
        Rmax = rowmax
        Cmin = colmin
        Cmax = colmax
    d = data[Rmin:Rmax + 1, Cmin:Cmax + 1]
    x = np.arange(Rmin, Rmax + 1)
    y = np.arange(Cmin, Cmax + 1)
    bcx = (d.sum(1) * x).sum() / d.sum()
    bcy = (d.sum(0) * y).sum() / d.sum()
    return bcx, bcy