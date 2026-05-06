def usearch_sort_by_abundance(
        fasta_filepath,
        output_filepath=None,
        sizein=True,
        sizeout=True,
        minsize=0,
        log_name="abundance_sort.log",
        usersort=False,
        HALT_EXEC=False,
        save_intermediate_files=False,
        remove_usearch_logs=False,
        working_dir=None):
    """ Sorts fasta file by abundance

    fasta_filepath = input fasta file, generally a dereplicated fasta
    output_filepath = output abundance sorted fasta filepath
    sizein = not defined in usearch helpstring
    sizeout = not defined in usearch helpstring
    minsize = minimum size of cluster to retain.
    log_name = string to specify log filename
    usersort = Use if not sorting by abundance or usearch will raise an error
    HALT_EXEC: Used for debugging app controller
    save_intermediate_files: Preserve all intermediate files created.
    """
    if not output_filepath:
        _, output_filepath = mkstemp(prefix='usearch_abundance_sorted',
                                     suffix='.fasta')

    log_filepath = join(
        working_dir,
        "minsize_" + str(minsize) + "_" + log_name)

    params = {}

    app = Usearch(params, WorkingDir=working_dir, HALT_EXEC=HALT_EXEC)

    if usersort:
        app.Parameters['--usersort'].on()

    if minsize:
        app.Parameters['--minsize'].on(minsize)

    if sizein:
        app.Parameters['--sizein'].on()

    if sizeout:
        app.Parameters['--sizeout'].on()

    data = {'--sortsize': fasta_filepath,
            '--output': output_filepath
            }

    if not remove_usearch_logs:
        data['--log'] = log_filepath

    # Can have no data following this filter step, which will raise an
    # application error, try to catch it here to raise meaningful message.

    try:
        app_result = app(data)
    except ApplicationError:
        raise ValueError('No data following filter steps, please check ' +
                         'parameter settings for usearch_qf.')

    return app_result, output_filepath