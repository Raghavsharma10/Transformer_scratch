def filter_mc_sharemem(filename, step_size, box_size, cores, shape, nslice=None, domask=True):
    """
    Calculate the background and noise images corresponding to the input file.
    The calculation is done via a box-car approach and uses multiple cores and shared memory.

    Parameters
    ----------
    filename : str
        Filename to be filtered.

    step_size : (int, int)
        Step size for the filter.

    box_size : (int, int)
        Box size for the filter.

    cores : int
        Number of cores to use. If None then use all available.

    nslice : int
        The image will be divided into this many horizontal stripes for processing.
        Default = None = equal to cores

    shape : (int, int)
        The shape of the image in the given file.

    domask : bool
        True(Default) = copy data mask to output.

    Returns
    -------
    bkg, rms : numpy.ndarray
        The interpolated background and noise images.
    """

    if cores is None:
        cores = multiprocessing.cpu_count()
    if (nslice is None) or (cores==1):
        nslice = cores

    img_y, img_x = shape
    # initialise some shared memory
    global ibkg
    # bkg = np.ctypeslib.as_ctypes(np.empty(shape, dtype=np.float32))
    # ibkg = multiprocessing.sharedctypes.Array(bkg._type_, bkg, lock=True)
    ibkg = multiprocessing.Array('f', img_y*img_x)

    global irms
    #rms = np.ctypeslib.as_ctypes(np.empty(shape, dtype=np.float32))
    #irms = multiprocessing.sharedctypes.Array(rms._type_, rms, lock=True)
    irms = multiprocessing.Array('f', img_y * img_x)

    logging.info("using {0} cores".format(cores))
    logging.info("using {0} stripes".format(nslice))

    if nslice > 1:
        # box widths should be multiples of the step_size, and not zero
        width_y = int(max(img_y/nslice/step_size[1], 1) * step_size[1])

        # locations of the box edges
        ymins = list(range(0, img_y, width_y))
        ymaxs = list(range(width_y, img_y, width_y))
        ymaxs.append(img_y)
    else:
        ymins = [0]
        ymaxs = [img_y]

    logging.debug("ymins {0}".format(ymins))
    logging.debug("ymaxs {0}".format(ymaxs))

    # create an event per stripe
    global bkg_events, mask_events
    bkg_events = [multiprocessing.Event() for _ in range(len(ymaxs))]
    mask_events = [multiprocessing.Event() for _ in range(len(ymaxs))]

    args = []
    for i, region in enumerate(zip(ymins, ymaxs)):
        args.append((filename, region, step_size, box_size, shape, domask, i))

    # start a new process for each task, hopefully to reduce residual memory use
    pool = multiprocessing.Pool(processes=cores, maxtasksperchild=1)
    try:
        # chunksize=1 ensures that we only send a single task to each process
        pool.map_async(_sf2, args, chunksize=1).get(timeout=10000000)
    except KeyboardInterrupt:
        logging.error("Caught keyboard interrupt")
        pool.close()
        sys.exit(1)
    pool.close()
    pool.join()
    bkg = np.reshape(np.array(ibkg[:], dtype=np.float32), shape)
    rms = np.reshape(np.array(irms[:], dtype=np.float32), shape)
    del ibkg, irms
    return bkg, rms