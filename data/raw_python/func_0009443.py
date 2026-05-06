def visualize(hog, grid=(10, 10), radCircle=None):
    '''
    visualize HOG as polynomial around cell center
        for [grid] * cells
    '''
    s0, s1, nang = hog.shape
    angles = np.linspace(0, np.pi, nang + 1)[:-1]
    # center of each sub array:
    cx, cy = s0 // (2 * grid[0]), s1 // (2 * grid[1])
    # max. radius of polynomial around cenetr:
    rx, ry = cx, cy
    # for drawing a position indicator (circle):
    if radCircle is None:
        radCircle = max(1, rx // 10)
    # output array:
    out = np.zeros((s0, s1), dtype=np.uint8)
    # point of polynomial:
    pts = np.empty(shape=(1, 2 * nang, 2), dtype=np.int32)
    # takes grid[0]*grid[1] sample HOG values:
    samplesHOG = subCell2DFnArray(hog, lambda arr: arr[cx, cy], grid)
    mxHOG = samplesHOG.max()
    # sub array slices:
    slices = list(subCell2DSlices(out, grid))
    m = 0
    for m, hhh in enumerate(samplesHOG.reshape(grid[0] * grid[1], nang)):
        hhmax = hhh.max()
        hh = hhh / hhmax
        sout = out[slices[m][2:4]]
        for n, (o, a) in enumerate(zip(hh, angles)):
            pts[0, n, 0] = cx + np.cos(a) * o * rx
            pts[0, n, 1] = cy + np.sin(a) * o * ry
            pts[0, n + nang, 0] = cx + np.cos(a + np.pi) * o * rx
            pts[0, n + nang, 1] = cy + np.sin(a + np.pi) * o * ry

        cv2.fillPoly(sout, pts, int(255 * hhmax / mxHOG))
        cv2.circle(sout, (cx, cy), radCircle, 0, thickness=-1)

    return out