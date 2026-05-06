def findbeam_gravity(data, mask):
    """Find beam center with the "gravity" method

    Inputs:
        data: scattering image
        mask: mask matrix

    Output:
        a vector of length 2 with the x (row) and y (column) coordinates
         of the origin, starting from 1
    """
    # for each row and column find the center of gravity
    data1 = data.copy()  # take a copy, because elements will be tampered with
    data1[mask == 0] = 0  # set masked elements to zero
    # vector of x (row) coordinates
    x = np.arange(data1.shape[0])
    # vector of y (column) coordinates
    y = np.arange(data1.shape[1])
    # two column vectors, both containing ones. The length of onex and
    # oney corresponds to length of x and y, respectively.
    onex = np.ones_like(x)
    oney = np.ones_like(y)
    # Multiply the matrix with x. Each element of the resulting column
    # vector will contain the center of gravity of the corresponding row
    # in the matrix, multiplied by the "weight". Thus: nix_i=sum_j( A_ij
    # * x_j). If we divide this by spamx_i=sum_j(A_ij), then we get the
    # center of gravity. The length of this column vector is len(y).
    nix = np.dot(x, data1)
    spamx = np.dot(onex, data1)
    # indices where both nix and spamx is nonzero.
    goodx = ((nix != 0) & (spamx != 0))
    # trim y, nix and spamx by goodx, eliminate invalid points.
    nix = nix[goodx]
    spamx = spamx[goodx]

    # now do the same for the column direction.
    niy = np.dot(data1, y)
    spamy = np.dot(data1, oney)
    goody = ((niy != 0) & (spamy != 0))
    niy = niy[goody]
    spamy = spamy[goody]
    # column coordinate of the center in each row will be contained in
    # ycent, the row coordinate of the center in each column will be
    # in xcent.
    ycent = nix / spamx
    xcent = niy / spamy
    # return the mean values as the centers.
    return [xcent.mean(), ycent.mean()]