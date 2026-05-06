def draw_flow(img, flow, step=16, dtype=uint8):
    """
    draws flow vectors on image
    this came from opencv/examples directory
    another way: http://docs.opencv.org/trunk/doc/py_tutorials/py_gui/py_drawing_functions/py_drawing_functions.html
    """
    maxval = iinfo(img.dtype).max

    # scaleFact = 1. #arbitary factor to make flow visible
    canno = (0, maxval, 0)  # green color
    h, w = img.shape[:2]
    y, x = mgrid[step//2:h:step, step//2:w:step].reshape(2, -1)
    fx, fy = flow[y, x].T
    # create line endpoints
    lines = vstack([x, y, (x+fx), (y+fy)]).T.reshape(-1, 2, 2)
    lines = int32(lines + 0.5)
    # create image
    if img.ndim == 2:  # assume gray
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:  # already RGB
        vis = img
    # draw line
    cv2.polylines(vis, lines, isClosed=False, color=canno, thickness=1, lineType=8)
    # draw filled green circles
    for (x1, y1), (x2, y2) in lines:
        cv2.circle(vis, center=(x1, y1), radius=1, color=canno, thickness=-1)
    return vis