def sigma_filter(filename, region, step_size, box_size, shape, domask, sid):
    """
    Calculate the background and rms for a sub region of an image. The results are
    written to shared memory - irms and ibkg.

    Parameters
    ----------
    filename : string
        Fits file to open

    region : list
        Region within the fits file that is to be processed. (row_min, row_max).

    step_size : (int, int)
        The filtering step size

    box_size : (int, int)
        The size of the box over which the filter is applied (each step).

    shape : tuple
        The shape of the fits image

    domask : bool
        If true then copy the data mask to the output.

    sid : int
        The stripe number

    Returns
    -------
    None
    """

    ymin, ymax = region
    logging.debug('rows {0}-{1} starting at {2}'.format(ymin, ymax, strftime("%Y-%m-%d %H:%M:%S", gmtime())))

    # cut out the region of interest plus 1/2 the box size, but clip to the image size
    data_row_min = max(0, ymin - box_size[0]//2)
    data_row_max = min(shape[0], ymax + box_size[0]//2)

    # Figure out how many axes are in the datafile
    NAXIS = fits.getheader(filename)["NAXIS"]

    with fits.open(filename, memmap=True) as a:
        if NAXIS == 2:
            data = a[0].section[data_row_min:data_row_max, 0:shape[1]]
        elif NAXIS == 3:
            data = a[0].section[0, data_row_min:data_row_max, 0:shape[1]]
        elif NAXIS == 4:
            data = a[0].section[0, 0, data_row_min:data_row_max, 0:shape[1]]
        else:
            logging.error("Too many NAXIS for me {0}".format(NAXIS))
            logging.error("fix your file to be more sane")
            raise Exception("Too many NAXIS")

    row_len = shape[1]

    logging.debug('data size is {0}'.format(data.shape))

    def box(r, c):
        """
        calculate the boundaries of the box centered at r,c
        with size = box_size
        """
        r_min = max(0, r - box_size[0] // 2)
        r_max = min(data.shape[0] - 1, r + box_size[0] // 2)
        c_min = max(0, c - box_size[1] // 2)
        c_max = min(data.shape[1] - 1, c + box_size[1] // 2)
        return r_min, r_max, c_min, c_max

    # set up a grid of rows/cols at which we will compute the bkg/rms
    rows = list(range(ymin-data_row_min, ymax-data_row_min, step_size[0]))
    rows.append(ymax-data_row_min)
    cols = list(range(0, shape[1], step_size[1]))
    cols.append(shape[1])

    # store the computed bkg/rms in this smaller array
    vals = np.zeros(shape=(len(rows),len(cols)))

    for i, row in enumerate(rows):
        for j, col in enumerate(cols):
            r_min, r_max, c_min, c_max = box(row, col)
            new = data[r_min:r_max, c_min:c_max]
            new = np.ravel(new)
            bkg, _ = sigmaclip(new, 3, 3)
            vals[i,j] = bkg

    # indices of all the pixels within our region
    gr, gc = np.mgrid[ymin-data_row_min:ymax-data_row_min, 0:shape[1]]

    logging.debug("Interpolating bkg to sharemem")
    ifunc = RegularGridInterpolator((rows, cols), vals)
    for i in range(gr.shape[0]):
        row = np.array(ifunc((gr[i], gc[i])), dtype=np.float32)
        start_idx = np.ravel_multi_index((ymin+i, 0), shape)
        end_idx = start_idx + row_len
        ibkg[start_idx:end_idx] = row  # np.ctypeslib.as_ctypes(row)
    del ifunc
    logging.debug(" ... done writing bkg")

    # signal that the bkg is done for this region, and wait for neighbours
    barrier(bkg_events, sid)

    logging.debug("{0} background subtraction".format(sid))
    for i in range(data_row_max - data_row_min):
        start_idx = np.ravel_multi_index((data_row_min + i, 0), shape)
        end_idx = start_idx + row_len
        data[i, :] = data[i, :] - ibkg[start_idx:end_idx]
    # reset/recycle the vals array
    vals[:] = 0

    for i, row in enumerate(rows):
        for j, col in enumerate(cols):
            r_min, r_max, c_min, c_max = box(row, col)
            new = data[r_min:r_max, c_min:c_max]
            new = np.ravel(new)
            _ , rms = sigmaclip(new, 3, 3)
            vals[i,j] = rms

    logging.debug("Interpolating rm to sharemem rms")
    ifunc = RegularGridInterpolator((rows, cols), vals)
    for i in range(gr.shape[0]):
        row = np.array(ifunc((gr[i], gc[i])), dtype=np.float32)
        start_idx = np.ravel_multi_index((ymin+i, 0), shape)
        end_idx = start_idx + row_len
        irms[start_idx:end_idx] = row  # np.ctypeslib.as_ctypes(row)
    del ifunc
    logging.debug(" .. done writing rms")

    if domask:
        barrier(mask_events, sid)
        logging.debug("applying mask")
        for i in range(gr.shape[0]):
            mask = np.where(np.bitwise_not(np.isfinite(data[i + ymin-data_row_min,:])))[0]
            for j in mask:
                idx = np.ravel_multi_index((i + ymin,j),shape)
                ibkg[idx] = np.nan
                irms[idx] = np.nan
        logging.debug(" ... done applying mask")
    logging.debug('rows {0}-{1} finished at {2}'.format(ymin, ymax, strftime("%Y-%m-%d %H:%M:%S", gmtime())))
    return