def _satdet_worker(work_queue, done_queue, sigma=2.0, low_thresh=0.1,
                   h_thresh=0.5, small_edge=60, line_len=200, line_gap=75,
                   percentile=(4.5, 93.0), buf=200):
    """Multiprocessing worker."""
    for fil, chip in iter(work_queue.get, 'STOP'):
        try:
            result = _detsat_one(
                fil, chip, sigma=sigma,
                low_thresh=low_thresh, h_thresh=h_thresh,
                small_edge=small_edge, line_len=line_len, line_gap=line_gap,
                percentile=percentile, buf=buf, plot=False, verbose=False)
        except Exception as e:
            retcode = False
            result = '{0}: {1}'.format(type(e), str(e))
        else:
            retcode = True
        done_queue.put((retcode, fil, chip, result))

    return True