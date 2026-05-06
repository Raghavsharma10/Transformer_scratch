def get_outliers(campaign, pipeline='everest2', sigma=5):
    '''
    Computes the number of outliers for a given `campaign`
    and a given `pipeline`.
    Stores the results in a file under "/missions/k2/tables/".

    :param int sigma: The sigma level at which to clip outliers. Default 5

    '''

    # Imports
    from .utils import GetK2Campaign
    client = k2plr.API()

    # Check pipeline
    assert pipeline.lower() in Pipelines, 'Invalid pipeline: `%s`.' % pipeline

    # Create file if it doesn't exist
    file = os.path.join(EVEREST_SRC, 'missions', 'k2',
                        'tables', 'c%02d_%s.out' % (int(campaign), pipeline))
    if not os.path.exists(file):
        open(file, 'a').close()

    # Get all EPIC stars
    stars = GetK2Campaign(campaign, epics_only=True)
    nstars = len(stars)

    # Remove ones we've done
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        done = np.loadtxt(file, dtype=float)
    if len(done):
        done = [int(s) for s in done[:, 0]]
    stars = list(set(stars) - set(done))
    n = len(done) + 1

    # Open the output file
    with open(file, 'a', 1) as outfile:

        # Loop over all to get the CDPP
        for EPIC in stars:

            # Progress
            sys.stdout.write('\rRunning target %d/%d...' % (n, nstars))
            sys.stdout.flush()
            n += 1

            # Get the number of outliers
            try:
                time, flux = get(EPIC, pipeline=pipeline, campaign=campaign)

                # Get the raw K2 data
                tpf = os.path.join(KPLR_ROOT, "data", "k2",
                                   "target_pixel_files",
                                   "%09d" % EPIC,
                                   "ktwo%09d-c%02d_lpd-targ.fits.gz"
                                   % (EPIC, campaign))
                if not os.path.exists(tpf):
                    client.k2_star(EPIC).get_target_pixel_files(fetch=True)
                with pyfits.open(tpf) as f:
                    k2_qual = np.array(f[1].data.field('QUALITY'), dtype=int)
                    k2_time = np.array(
                        f[1].data.field('TIME'), dtype='float64')
                    mask = []
                    for b in [1, 2, 3, 4, 5, 6, 7, 8, 9,
                              11, 12, 13, 14, 16, 17]:
                        mask += list(np.where(k2_qual & 2 ** (b - 1))[0])
                    mask = np.array(sorted(list(set(mask))))

                # Fill in missing cadences, if any
                tol = 0.005
                if not ((len(time) == len(k2_time)) and (np.abs(time[0]
                        - k2_time[0]) < tol) and (np.abs(time[-1]
                                                  - k2_time[-1]) < tol)):
                    ftmp = np.zeros_like(k2_time) * np.nan
                    j = 0
                    for i, t in enumerate(k2_time):
                        if np.abs(time[j] - t) < tol:
                            ftmp[i] = flux[j]
                            j += 1
                            if j == len(time) - 1:
                                break
                    flux = ftmp

                # Remove flagged cadences
                flux = np.delete(flux, mask)

                # Remove nans
                nanmask = np.where(np.isnan(flux))[0]
                flux = np.delete(flux, nanmask)

                # Iterative sigma clipping
                inds = np.array([], dtype=int)
                m = 1
                while len(inds) < m:
                    m = len(inds)
                    f = SavGol(np.delete(flux, inds))
                    med = np.nanmedian(f)
                    MAD = 1.4826 * np.nanmedian(np.abs(f - med))
                    inds = np.append(inds, np.where(
                        (f > med + sigma * MAD) | (f < med - sigma * MAD))[0])
                nout = len(inds)
                ntot = len(flux)

            except (urllib.error.HTTPError, urllib.error.URLError,
                    TypeError, ValueError, IndexError):
                print("{:>09d} {:>5d} {:>5d}".format(
                    EPIC, -1, -1), file=outfile)
                continue

            # Log to file
            print("{:>09d} {:>5d} {:>5d}".format(
                EPIC, nout, ntot), file=outfile)