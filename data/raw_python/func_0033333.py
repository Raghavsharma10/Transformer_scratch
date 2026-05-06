def vsearch_chimera_filter_de_novo(
    fasta_filepath,
    working_dir,
    output_chimeras=True,
    output_nonchimeras=True,
    output_alns=False,
    output_tabular=False,
    log_name="vsearch_uchime_de_novo_chimera_filtering.log",
    HALT_EXEC=False):
    """ Detect chimeras present in the fasta-formatted filename,
        without external references (i.e. de novo). Automatically
        sort the sequences in filename by decreasing abundance
        beforehand. Output chimeras and non-chimeras to FASTA files
        and/or 3-way global alignments and/or tabular output.

        Parameters
        ----------

        fasta_filepath : string
           input fasta file (dereplicated fasta with pattern
           [>;]size=integer[;] in the fasta header)
        working_dir : string
           directory path for all output files
        output_chimeras : boolean, optional
           output chimeric sequences to file, in fasta format
        output_nonchimeras : boolean, optional
           output nonchimeric sequences to file, in fasta format
        output_alns : boolean, optional
           output 3-way global alignments (parentA, parentB, chimera)
           in human readable format to file
        output_tabular : boolean, optional
           output results using the uchime tab-separated format of
           18 fields (see Vsearch user manual)
        HALT_EXEC : boolean, optional
           used for debugging app controller

        Return
        ------

        output_chimera_filepath : string
           filepath to chimeric fasta sequences
        output_non_chimera_filepath : string
           filepath to nonchimeric fasta sequences
        output_alns_filepath : string
           filepath to chimeric sequences alignment
           file
        output_tabular_filepath : string
           filepath to chimeric sequences tabular
           output file
        log_filepath : string
           filepath to log file
    """

    app = Vsearch(WorkingDir=working_dir, HALT_EXEC=HALT_EXEC)

    if not (output_chimeras or
            output_nonchimeras or
            output_alns or
            output_tabular):
        raise ValueError("At least one output format (output_chimeras,"
                         "output_nonchimeras, output_alns, output_tabular)"
                         "must be selected")

    output_chimera_filepath = None
    output_non_chimera_filepath = None
    output_alns_filepath = None
    output_tabular_filepath = None

    # set output filepaths
    if output_chimeras:
        output_chimera_filepath = join(working_dir, 'uchime_chimeras.fasta')
        app.Parameters['--chimeras'].on(output_chimera_filepath)
    if output_nonchimeras:
        output_non_chimera_filepath = join(working_dir,
                                           'uchime_non_chimeras.fasta')
        app.Parameters['--nonchimeras'].on(output_non_chimera_filepath)
    if output_alns:
        output_alns_filepath = join(working_dir, 'uchime_alignments.txt')
        app.Parameters['--uchimealns'].on(output_alns_filepath)
    if output_tabular:
        output_tabular_filepath = join(working_dir, 'uchime_tabular.txt')
        app.Parameters['--uchimeout'].on(output_tabular_filepath)
    log_filepath = join(working_dir, log_name)

    app.Parameters['--uchime_denovo'].on(fasta_filepath)
    app.Parameters['--log'].on(log_filepath)

    app_result = app()

    return output_chimera_filepath, output_non_chimera_filepath,\
        output_alns_filepath, output_tabular_filepath, log_filepath