def warp_image_by_corner_points_projection(corner_points, image):
    """Given corner points of a Sudoku, warps original selection to a square image.

    :param corner_points:
    :type: corner_points: list
    :param image:
    :type image:
    :return:
    :rtype:

    """
    # Clarify by storing in named variables.
    top_left, top_right, bottom_left, bottom_right = np.array(corner_points)

    top_edge = np.linalg.norm(top_right - top_left)
    bottom_edge = np.linalg.norm(bottom_right - bottom_left)
    left_edge = np.linalg.norm(top_left - bottom_left)
    right_edge = np.linalg.norm(top_right - bottom_right)

    L = int(np.ceil(max([top_edge, bottom_edge, left_edge, right_edge])))
    src = np.array([top_left, top_right, bottom_left, bottom_right])
    dst = np.array([[0, 0], [L - 1, 0], [0, L - 1], [L - 1, L - 1]])

    tr = ProjectiveTransform()
    tr.estimate(dst, src)
    warped_image = warp(image, tr, output_shape=(L, L))
    out = resize(warped_image, (500, 500))

    return out