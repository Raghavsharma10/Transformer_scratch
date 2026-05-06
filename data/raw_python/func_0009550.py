def gridLinesFromVertices(edges, nCells, subgrid=None, dtype=float):
    """
    ###TODO  REDO TXT

    OPTIONAL:
    subgrid = ([x],[y]) --> relative positions
        e.g. subgrid = ( (0.3,0.7), () )
             --> two subgrid lines in x - nothing in y

    Returns: 
        horiz,vert -> arrays of (x,y) poly-lines


    if subgrid != None, Returns:
            horiz,vert, subhoriz, subvert


    #######
    creates a regular 2d grid from given edge points (4*(x0,y0))
    and number of cells in x and y

    Returns:
        tuple(4lists): horizontal and vertical lines as (x0,y0,x1,y1)
    """

    nx, ny = nCells

    y, x = np.mgrid[0.:ny + 1, 0.:nx + 1]

    src = np.float32([[0, 0], [nx, 0], [nx, ny], [0, ny]])
    dst = sortCorners(edges).astype(np.float32)

    homography = cv2.getPerspectiveTransform(src, dst)

    pts = np.float32((x.flatten(), y.flatten())).T
    pts = pts.reshape(1, *pts.shape)

    pts2 = cv2.perspectiveTransform(pts, homography)[0]

    horiz = pts2.reshape(ny + 1, nx + 1, 2)
    vert = np.swapaxes(horiz, 0, 1)

    subh, subv = [], []
    if subgrid is not None:
        sh, sv = subgrid

        if len(sh):
            subh = np.empty(shape=(ny * len(sh), nx + 1, 2), dtype=np.float32)
            last_si = 0
            for n, si in enumerate(sh):
                spts = pts[:, :-(nx + 1)]
                spts[..., 1] += si - last_si
                last_si = si
                spts2 = cv2.perspectiveTransform(spts, homography)[0]
                subh[n::len(sh)] = spts2.reshape(ny, nx + 1, 2)
        if len(sv):
            subv = np.empty(shape=(ny + 1, nx * len(sv), 2), dtype=np.float32)
            last_si = 0
            sspts = pts.reshape(1, ny + 1, nx + 1, 2)
            sspts = sspts[:, :, :-1]

            sspts = sspts.reshape(1, (ny + 1) * nx, 2)
            for n, si in enumerate(sv):
                sspts[..., 0] += si - last_si
                last_si = si
                spts2 = cv2.perspectiveTransform(sspts, homography)[0]
                subv[:, n::len(sv)] = spts2.reshape(ny + 1, nx, 2)
            subv = np.swapaxes(subv, 0, 1)
    return [horiz, vert, subh, subv]