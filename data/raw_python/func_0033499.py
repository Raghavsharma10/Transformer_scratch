def usearch_fasta_sort_from_filepath(
        fasta_filepath,
        output_filepath=None,
        log_name="sortlen.log",
        HALT_EXEC=False,
        save_intermediate_files=False,
        remove_usearch_logs=False,
        working_dir=None):
    """Generates sorted fasta file via usearch --mergesort.

    fasta_filepath: filepath to input fasta file
    output_filepath: filepath for output sorted fasta file.
    log_name: string to specify log filename
    HALT_EXEC: Used for debugging app controller
    save_intermediate_files: Preserve all intermediate files created."""
    if not output_filepath:
        _, output_filepath = mkstemp(prefix='usearch_fasta_sort',
                                     suffix='.fasta')

    log_filepath = join(working_dir, log_name)

    params = {}

    app = Usearch(params, WorkingDir=working_dir, HALT_EXEC=HALT_EXEC)

    data = {'--mergesort': fasta_filepath,
            '--output': output_filepath,
            }

    if not remove_usearch_logs:
        data['--log'] = log_filepath

    app_result = app(data)

    return app_result, output_filepath