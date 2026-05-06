def usearch_chimera_filter_de_novo(
        fasta_filepath,
        output_chimera_filepath=None,
        output_non_chimera_filepath=None,
        abundance_skew=2.0,
        log_name="uchime_de_novo_chimera_filtering.log",
        usersort=False,
        HALT_EXEC=False,
        save_intermediate_files=False,
        remove_usearch_logs=False,
        working_dir=None):
    """ Chimera filter de novo, output chimeras and non-chimeras to fastas

    fasta_filepath = input fasta file, generally a dereplicated fasta
    output_chimera_filepath = output chimera filepath
    output_non_chimera_filepath = output non chimera filepath
    abundance_skew = abundance skew setting for de novo filtering.
    usersort = Enable if input fasta not sorted by length purposefully, lest
     usearch will raise an error.
    HALT_EXEC: Used for debugging app controller
    save_intermediate_files: Preserve all intermediate files created.
    """
    if not output_chimera_filepath:
        _, output_chimera_filepath = mkstemp(prefix='uchime_chimeras_',
                                             suffix='.fasta')

    if not output_non_chimera_filepath:
        _, output_non_chimera_filepath = mkstemp(prefix='uchime_non_chimeras_',
                                                 suffix='.fasta')

    log_filepath = join(working_dir, log_name)

    params = {'--abskew': abundance_skew}

    app = Usearch(params, WorkingDir=working_dir, HALT_EXEC=HALT_EXEC)

    if usersort:
        app.Parameters['--usersort'].on()

    data = {'--uchime': fasta_filepath,
            '--chimeras': output_chimera_filepath,
            '--nonchimeras': output_non_chimera_filepath
            }

    if not remove_usearch_logs:
        data['--log'] = log_filepath

    app_result = app(data)

    if not save_intermediate_files:
        remove_files([output_chimera_filepath])

    return app_result, output_non_chimera_filepath