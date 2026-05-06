def cli(sequencepath, report, refseq_database):
    """
    Pass command line arguments to, and run the feature extraction functions
    """
    main(sequencepath, report, refseq_database, num_threads=multiprocessing.cpu_count())