def detsat(searchpattern, chips=[1, 4], n_processes=4, sigma=2.0,
           low_thresh=0.1, h_thresh=0.5, small_edge=60, line_len=200,
           line_gap=75, percentile=(4.5, 93.0), buf=200, plot=False,
           verbose=True):
    """Find satellite trails in the given images and extensions.
    The trails are calculated using Probabilistic Hough Transform.

    .. note::

        The trail endpoints found here are crude approximations.
        Use :func:`make_mask` to create the actual DQ mask for the trail(s)
        of interest.

    Parameters
    ----------
    searchpattern : str
        Search pattern for input FITS images, as accepted by
        :py:func:`glob.glob`.

    chips : list
        List of extensions for science data, as accepted by
        ``astropy.io.fits``.
        The default values of ``[1, 4]`` are tailored for ACS/WFC.

    n_processes : int
        Number of processes for multiprocessing, which is only useful
        if you are processing a lot of images or extensions.
        If 1 is given, no multiprocessing is done.

    sigma : float, optional
        The size of a Gaussian filter to use before edge detection.
        The default is 2, which is good for almost all images.

    low_thresh : float, optional
        The lower threshold for hysteresis linking of edge pieces.
        This should be between 0 and 1, and less than ``h_thresh``.

    h_thresh : float, optional
        The upper threshold for hysteresis linking of edge pieces.
        This should be between 0 and 1, and greater than ``low_thresh``.

    small_edge : int, optional
        Size of perimeter of small objects to remove in edge image.
        This significantly reduces noise before doing Hough Transform.
        If it is set too high, you will remove the edge of the
        satellite you are trying to find.

    line_len : int, optional
        Minimum line length for Probabilistic Hough Transform to fit.

    line_gap : int, optional
        The largest gap in points allowed for the Probabilistic
        Hough Transform.

    percentile : tuple of float, optional
        The percent boundaries to scale the image to before
        creating edge image.

    buf : int, optional
        How close to the edge of the image the satellite trail has to
        be to be considered a trail.

    plot : bool, optional
        Make plots of edge image, Hough space transformation, and
        rescaled image. This is only applicable if ``n_processes=1``.

    verbose : bool, optional
        Print extra information to the terminal, mostly for debugging.
        In multiprocessing mode, info from individual process is not printed.

    Returns
    -------
    results : dict
        Dictionary mapping ``(filename, ext)`` to an array of endpoints of
        line segments in the format of ``[[x0, y0], [x1, y1]]`` (if found) or
        an empty array (if not). These are the segments that have been
        identified as making up part of a satellite trail.

    errors : dict
        Dictionary mapping ``(filename, ext)`` to the error message explaining
        why processing failed.

    Raises
    ------
    ImportError
        Missing scipy or skimage>=0.11 packages.

    """
    if not HAS_OPDEP:
        raise ImportError('Missing scipy or skimage>=0.11 packages')

    if verbose:
        t_beg = time.time()

    files = glob.glob(searchpattern)
    n_files = len(files)
    n_chips = len(chips)
    n_tot = n_files * n_chips
    n_cpu = multiprocessing.cpu_count()
    results = {}
    errors = {}

    if verbose:
        print('{0} file(s) found...'.format(n_files))

    # Nothing to do
    if n_files < 1 or n_chips < 1:
        return results, errors

    # Adjust number of processes
    if n_tot < n_processes:
        n_processes = n_tot
    if n_processes > n_cpu:
        n_processes = n_cpu

    # No multiprocessing
    if n_processes == 1:
        for fil in files:
            for chip in chips:
                if verbose:
                    print('\nProcessing {0}[{1}]...'.format(fil, chip))

                key = (fil, chip)
                try:
                    result = _detsat_one(
                        fil, chip, sigma=sigma,
                        low_thresh=low_thresh, h_thresh=h_thresh,
                        small_edge=small_edge, line_len=line_len,
                        line_gap=line_gap, percentile=percentile, buf=buf,
                        plot=plot, verbose=verbose)
                except Exception as e:
                    errmsg = '{0}: {1}'.format(type(e), str(e))
                    errors[key] = errmsg
                    if verbose:
                        print(errmsg)
                else:
                    results[key] = result
        if verbose:
            print()

    # Multiprocessing.
    # The work queue is for things that need to be done and is shared by all
    # processes. When a worker finishes, its output is put into done queue.
    else:
        if verbose:
            print('Using {0} processes'.format(n_processes))

        work_queue = Queue()
        done_queue = Queue()
        processes = []

        for fil in files:
            for chip in chips:
                work_queue.put((fil, chip))

        for w in range(n_processes):
            p = Process(
                target=_satdet_worker, args=(work_queue, done_queue), kwargs={
                    'sigma': sigma, 'low_thresh': low_thresh,
                    'h_thresh': h_thresh, 'small_edge': small_edge,
                    'line_len': line_len, 'line_gap': line_gap,
                    'percentile': percentile, 'buf': buf})
            p.start()
            processes.append(p)
            work_queue.put('STOP')

        for p in processes:
            p.join()

        done_queue.put('STOP')

        # return a dictionary of lists
        for status in iter(done_queue.get, 'STOP'):
            key = (status[1], status[2])
            if status[0]:  # Success
                results[key] = status[3]
            else:  # Failed
                errors[key] = status[3]

        if verbose:
            if len(results) > 0:
                print('Number of trail segment(s) found:')
            for key in sorted(results):
                print('  {0}[{1}]: {2}'.format(
                    key[0], key[1], len(results[key])))
            if len(errors) > 0:
                print('These have errors:')
            for key in sorted(errors):
                print('  {0}[{1}]'.format(key[0], key[1]))

    if verbose:
        t_end = time.time()
        print('Total run time: {0} s'.format(t_end - t_beg))

    return results, errors