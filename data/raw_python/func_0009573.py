def alignImageAlongLine(img, line, height=15, length=None,
                        zoom=1, fast=False, borderValue=0):
    '''
    return a sub image aligned along given line

    @param img -  numpy.2darray input image to get subimage from
    @param line - list of 2 points [x0,y0,x1,y1])
    @param height - height of output array in y
    @param length - width of output array
    @param zoom - zoom factor
    @param fast - speed up calculation using nearest neighbour interpolation
    @returns transformed image as numpy.2darray with found line as in the middle
    '''

    height = int(round(height))
    if height % 2 == 0:  # ->is even number
        height += 1  # only take uneven numbers to have line in middle
    if length is None:
        length = int(round(ln.length(line)))
    hh = (height - 1)
    ll = (length - 1)

    # end points of the line:
    p0 = np.array(line[0:2], dtype=float)
    p1 = np.array(line[2:], dtype=float)
    # p2 is above middle of p0,p1:
    norm = np.array(ln.normal(line))
    if not ln.isHoriz(line):
        norm *= -1

    p2 = (p0 + p1) * 0.5 + norm * hh * 0.5
    middleY = hh / 2
    pp0 = [0, middleY]
    pp1 = [ll, middleY]
    pp2 = [ll * 0.5, hh]

    pts1 = np.array([p0, p1, p2], dtype=np.float32)
    pts2 = np.array([pp0, pp1, pp2], dtype=np.float32)

    if zoom != 1:
        length = int(round(length * zoom))
        height = int(round(height * zoom))
        pts2 *= zoom

    # TRANSFORM:
    M = cv2.getAffineTransform(pts1, pts2)
    dst = cv2.warpAffine(
        img, M, (length, height),
        flags=cv2.INTER_NEAREST if fast else cv2.INTER_LINEAR,
        borderValue=borderValue)
    return dst