def resize_to_shape(data, shape, zoom=None, mode="nearest", order=0):
    """
    Function resize input data to specific shape.
    :param data: input 3d array-like data
    :param shape: shape of output data
    :param zoom: zoom is used for back compatibility
    :mode: default is 'nearest'
    """
    # @TODO remove old code in except part
    # TODO use function from library in future

    try:
        # rint 'pred vyjimkou'
        # aise Exception ('test without skimage')
        # rint 'za vyjimkou'
        import skimage
        import skimage.transform

        # Now we need reshape  seeds and segmentation to original size

        # with warnings.catch_warnings():
        #     warnings.filterwarnings("ignore", ".*'constant', will be changed to.*")
        segm_orig_scale = skimage.transform.resize(
            data, shape, order=0, preserve_range=True, mode="reflect"
        )

        segmentation = segm_orig_scale
        logger.debug("resize to orig with skimage")
    except:
        if zoom is None:
            zoom = shape / np.asarray(data.shape).astype(np.double)
        segmentation = resize_to_shape_with_zoom(
            data, zoom=zoom, mode=mode, order=order
        )

    return segmentation