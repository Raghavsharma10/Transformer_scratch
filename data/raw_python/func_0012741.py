def runIDFs(jobs, processors=1):
    """Wrapper for run() to be used when running IDF5 runs in parallel.

    Parameters
    ----------
    jobs : iterable
        A list or generator made up of an IDF5 object and a kwargs dict
        (see `run_functions.run` for valid keywords).
    processors : int, optional
        Number of processors to run on (default: 1). If 0 is passed then
        the process will run on all CPUs, -1 means one less than all CPUs, etc.

    """
    if processors <= 0:
        processors = max(1, mp.cpu_count() - processors)

    shutil.rmtree("multi_runs", ignore_errors=True)
    os.mkdir("multi_runs")

    prepared_runs = (prepare_run(run_id, run_data) for run_id, run_data in enumerate(jobs))
    try:
        pool = mp.Pool(processors)
        pool.map(multirunner, prepared_runs)
        pool.close()
    except NameError:
        # multiprocessing not present so pass the jobs one at a time
        for job in prepared_runs:
            multirunner([job])
    shutil.rmtree("multi_runs", ignore_errors=True)