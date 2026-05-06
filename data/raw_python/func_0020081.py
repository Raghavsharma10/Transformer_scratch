def get_cdpp(campaign, pipeline='everest2'):
    '''
    Computes the CDPP for a given `campaign` and a given `pipeline`.
    Stores the results in a file under "/missions/k2/tables/".

    '''

    # Imports
    from .k2 import CDPP
    from .utils import GetK2Campaign

    # Check pipeline
    assert pipeline.lower() in Pipelines, 'Invalid pipeline: `%s`.' % pipeline

    # Create file if it doesn't exist
    file = os.path.join(EVEREST_SRC, 'missions', 'k2',
                        'tables', 'c%02d_%s.cdpp' % (int(campaign), pipeline))
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

            # Get the CDPP
            try:
                _, flux = get(EPIC, pipeline=pipeline, campaign=campaign)
                mask = np.where(np.isnan(flux))[0]
                flux = np.delete(flux, mask)
                cdpp = CDPP(flux)
            except (urllib.error.HTTPError, urllib.error.URLError,
                    TypeError, ValueError, IndexError):
                print("{:>09d} {:>15.3f}".format(EPIC, 0), file=outfile)
                continue

            # Log to file
            print("{:>09d} {:>15.3f}".format(EPIC, cdpp), file=outfile)