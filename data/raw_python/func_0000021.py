def crinfo_from_specific_data(data, margin=0):
    """
    Create crinfo of minimum orthogonal nonzero block in input data.

    :param data: input data
    :param margin: add margin to minimum block
    :return:
    """
    # hledáme automatický ořez, nonzero dá indexy
    logger.debug("crinfo")
    logger.debug(str(margin))
    nzi = np.nonzero(data)
    logger.debug(str(nzi))

    if np.isscalar(margin):
        margin = [margin] * 3

    x1 = np.min(nzi[0]) - margin[0]
    x2 = np.max(nzi[0]) + margin[0] + 1
    y1 = np.min(nzi[1]) - margin[0]
    y2 = np.max(nzi[1]) + margin[0] + 1
    z1 = np.min(nzi[2]) - margin[0]
    z2 = np.max(nzi[2]) + margin[0] + 1

    # ošetření mezí polí
    if x1 < 0:
        x1 = 0
    if y1 < 0:
        y1 = 0
    if z1 < 0:
        z1 = 0

    if x2 > data.shape[0]:
        x2 = data.shape[0] - 1
    if y2 > data.shape[1]:
        y2 = data.shape[1] - 1
    if z2 > data.shape[2]:
        z2 = data.shape[2] - 1

    # ořez
    crinfo = [[x1, x2], [y1, y2], [z1, z2]]
    return crinfo