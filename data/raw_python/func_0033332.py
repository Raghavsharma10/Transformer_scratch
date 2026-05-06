def vsearch_sort_by_abundance(
    fasta_filepath,
    output_filepath,
    working_dir=None,
    minsize=None,
    maxsize=None,
    log_name="abundance_sort.log",
    HALT_EXEC=False):
    """ Fasta entries are sorted by decreasing abundance
        (Fasta entries are assumed to be dereplicated with
        the pattern "[>;]size=integer[;]" present in the
        read label, ex. use function vsearch_dereplicate_exact_seqs
        prior to calling this function)

        Parameters
        ----------

        fasta_filepath : string
           input fasta file (dereplicated fasta)
        output_filepath : string
           output filepath for the sorted sequences in fasta format
        working_dir : string, optional
           working directory to store intermediate files
        minsize : integer, optional
           discard sequences with an abundance value smaller than
           minsize
        maxsize : integer, optional
           discard sequences with an abundance value greater than
           maxsize
        log_name : string, optional
           log filename
        HALT_EXEC : boolean, optional
           used for debugging app controller

        Return
        ------

        output_filepath : string
           filepath to sorted fasta file
        log_filepath : string
           filepath to log file
    """

    # set working dir to same directory as the output
    # file (if not provided)
    if not working_dir:
        working_dir = dirname(output_filepath)

    app = Vsearch(WorkingDir=working_dir, HALT_EXEC=HALT_EXEC)

    log_filepath = join(working_dir, log_name)

    if minsize:
        app.Parameters['--minsize'].on(minsize)

    if maxsize:
        app.Parameters['--maxsize'].on(maxsize)

    app.Parameters['--sortbysize'].on(fasta_filepath)
    app.Parameters['--output'].on(output_filepath)
    app.Parameters['--log'].on(log_filepath)

    app_result = app()

    return output_filepath, log_filepath