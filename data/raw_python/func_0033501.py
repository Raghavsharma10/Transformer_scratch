def usearch_cluster_error_correction(
        fasta_filepath,
        output_filepath=None,
        output_uc_filepath=None,
        percent_id_err=0.97,
        sizein=True,
        sizeout=True,
        w=64,
        slots=16769023,
        maxrejects=64,
        log_name="usearch_cluster_err_corrected.log",
        usersort=False,
        HALT_EXEC=False,
        save_intermediate_files=False,
        remove_usearch_logs=False,
        working_dir=None):
    """ Cluster for err. correction at percent_id_err, output consensus fasta

    fasta_filepath = input fasta file, generally a dereplicated fasta
    output_filepath = output error corrected fasta filepath
    percent_id_err = minimum identity percent.
    sizein = not defined in usearch helpstring
    sizeout = not defined in usearch helpstring
    w = Word length for U-sorting
    slots = Size of compressed index table. Should be prime, e.g. 40000003.
     Should also specify --w, typical is --w 16 or --w 32.
    maxrejects = Max rejected targets, 0=ignore, default 32.
    log_name = string specifying output log name
    usersort = Enable if input fasta not sorted by length purposefully, lest
     usearch will raise an error.
    HALT_EXEC: Used for debugging app controller
    save_intermediate_files: Preserve all intermediate files created.
    """
    if not output_filepath:
        _, output_filepath = mkstemp(prefix='usearch_cluster_err_corrected',
                                     suffix='.fasta')

    log_filepath = join(working_dir, log_name)

    params = {'--sizein': sizein,
              '--sizeout': sizeout,
              '--id': percent_id_err,
              '--w': w,
              '--slots': slots,
              '--maxrejects': maxrejects}

    app = Usearch(params, WorkingDir=working_dir, HALT_EXEC=HALT_EXEC)

    if usersort:
        app.Parameters['--usersort'].on()

    data = {'--cluster': fasta_filepath,
            '--consout': output_filepath
            }

    if not remove_usearch_logs:
        data['--log'] = log_filepath

    if output_uc_filepath:
        data['--uc'] = output_uc_filepath

    app_result = app(data)

    return app_result, output_filepath