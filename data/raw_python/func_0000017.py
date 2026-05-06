def seed_zoom(seeds, zoom):
    """
    Smart zoom for sparse matrix. If there is resize to bigger resolution
    thin line of label could be lost. This function prefers labels larger
    then zero. If there is only one small voxel in larger volume with zeros
    it is selected.
    """
    # import scipy
    # loseeds=seeds
    labels = np.unique(seeds)
    # remove first label - 0
    labels = np.delete(labels, 0)
    # @TODO smart interpolation for seeds in one block
    #        loseeds = scipy.ndimage.interpolation.zoom(
    #            seeds, zoom, order=0)
    loshape = np.ceil(np.array(seeds.shape) * 1.0 / zoom).astype(np.int)
    loseeds = np.zeros(loshape, dtype=np.int8)
    loseeds = loseeds.astype(np.int8)
    for label in labels:
        a, b, c = np.where(seeds == label)
        loa = np.round(a // zoom)
        lob = np.round(b // zoom)
        loc = np.round(c // zoom)
        # loseeds = np.zeros(loshape)

        loseeds[loa, lob, loc] += label
        # this is to detect conflict seeds
        loseeds[loseeds > label] = 100

    # remove conflict seeds
    loseeds[loseeds > 99] = 0

    # import py3DSeedEditor
    # ped = py3DSeedEditor.py3DSeedEditor(loseeds)
    # ped.show()

    return loseeds