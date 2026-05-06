def uncrop(data, crinfo, orig_shape, resize=False, outside_mode="constant", cval=0):
    """
    Put some boundary to input image.


    :param data: input data
    :param crinfo: array with minimum and maximum index along each axis
        [[minX, maxX],[minY, maxY],[minZ, maxZ]]. If crinfo is None, the whole input image is placed into [0, 0, 0].
        If crinfo is just series of three numbers, it is used as an initial point for input image placement.
    :param orig_shape: shape of uncropped image
    :param resize: True or False (default). Usefull if the data.shape does not fit to crinfo shape.
    :param outside_mode: 'constant', 'nearest'
    :return:
    """

    if crinfo is None:
        crinfo = list(zip([0] * data.ndim, orig_shape))
    elif np.asarray(crinfo).size == data.ndim:
        crinfo = list(zip(crinfo, np.asarray(crinfo) + data.shape))

    crinfo = fix_crinfo(crinfo)
    data_out = np.ones(orig_shape, dtype=data.dtype) * cval

    # print 'uncrop ', crinfo
    # print orig_shape
    # print data.shape
    if resize:
        data = resize_to_shape(data, crinfo[:, 1] - crinfo[:, 0])

    startx = np.round(crinfo[0][0]).astype(int)
    starty = np.round(crinfo[1][0]).astype(int)
    startz = np.round(crinfo[2][0]).astype(int)

    data_out[
        # np.round(crinfo[0][0]).astype(int):np.round(crinfo[0][1]).astype(int)+1,
        # np.round(crinfo[1][0]).astype(int):np.round(crinfo[1][1]).astype(int)+1,
        # np.round(crinfo[2][0]).astype(int):np.round(crinfo[2][1]).astype(int)+1
        startx : startx + data.shape[0],
        starty : starty + data.shape[1],
        startz : startz + data.shape[2],
    ] = data

    if outside_mode == "nearest":
        # for ax in range(data.ndims):
        # ax = 0

        # copy border slice to pixels out of boundary - the higher part
        for ax in range(data.ndim):
            # the part under the crop
            start = np.round(crinfo[ax][0]).astype(int)
            slices = [slice(None), slice(None), slice(None)]
            slices[ax] = start
            repeated_slice = np.expand_dims(data_out[slices], ax)
            append_sz = start
            if append_sz > 0:
                tile0 = np.repeat(repeated_slice, append_sz, axis=ax)
                slices = [slice(None), slice(None), slice(None)]
                slices[ax] = slice(None, start)
                # data_out[start + data.shape[ax] : , :, :] = tile0
                data_out[slices] = tile0
                # plt.imshow(np.squeeze(repeated_slice))
                # plt.show()

            # the part over the crop
            start = np.round(crinfo[ax][0]).astype(int)
            slices = [slice(None), slice(None), slice(None)]
            slices[ax] = start + data.shape[ax] - 1
            repeated_slice = np.expand_dims(data_out[slices], ax)
            append_sz = data_out.shape[ax] - (start + data.shape[ax])
            if append_sz > 0:
                tile0 = np.repeat(repeated_slice, append_sz, axis=ax)
                slices = [slice(None), slice(None), slice(None)]
                slices[ax] = slice(start + data.shape[ax], None)
                # data_out[start + data.shape[ax] : , :, :] = tile0
                data_out[slices] = tile0
                # plt.imshow(np.squeeze(repeated_slice))
                # plt.show()

    return data_out