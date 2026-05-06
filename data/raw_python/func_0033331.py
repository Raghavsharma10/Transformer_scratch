def vsearch_dereplicate_exact_seqs(
    fasta_filepath,
    output_filepath,
    output_uc=False,
    working_dir=None,
    strand="both",
    maxuniquesize=None,
    minuniquesize=None,
    sizein=False,
    sizeout=True,
    log_name="derep.log",
    HALT_EXEC=False):
    """ Generates clusters and fasta file of
        dereplicated subsequences

        Parameters
        ----------

        fasta_filepath : string
           input filepath of fasta file to be dereplicated
        output_filepath : string
           write the dereplicated sequences to output_filepath
        working_dir : string, optional
           directory path for storing intermediate output
        output_uc : boolean, optional
           uutput dereplication results in a file using a
           uclust-like format
        strand : string, optional
           when searching for strictly identical sequences,
           check the 'strand' only (default: both) or
           check the plus strand only
        maxuniquesize : integer, optional
           discard sequences with an abundance value greater
           than maxuniquesize
        minuniquesize : integer, optional
           discard sequences with an abundance value smaller
           than integer
        sizein : boolean, optional
           take into account the abundance annotations present in
           the input fasta file,  (search for the pattern
           "[>;]size=integer[;]" in sequence headers)
        sizeout : boolean, optional
           add abundance annotations to the output fasta file
           (add the pattern ";size=integer;" to sequence headers)
        log_name : string, optional
           specifies log filename
        HALT_EXEC : boolean, optional
           used for debugging app controller

        Return
        ------

        output_filepath : string
           filepath to dereplicated fasta file
        uc_filepath : string
           filepath to dereplication results in uclust-like format
        log_filepath : string
           filepath to log file
    """

    # write all vsearch output files to same directory
    # as output_filepath if working_dir is not specified
    if not working_dir:
        working_dir = dirname(abspath(output_filepath))

    app = Vsearch(WorkingDir=working_dir, HALT_EXEC=HALT_EXEC)

    log_filepath = join(working_dir, log_name)
    uc_filepath = None
    if output_uc:
        root_name = splitext(abspath(output_filepath))[0]
        uc_filepath = join(working_dir, '%s.uc' % root_name)
        app.Parameters['--uc'].on(uc_filepath)

    if maxuniquesize:
        app.Parameters['--maxuniquesize'].on(maxuniquesize)
    if minuniquesize:
        app.Parameters['--minuniquesize'].on(minuniquesize)
    if sizein:
        app.Parameters['--sizein'].on()
    if sizeout:
        app.Parameters['--sizeout'].on()
    if (strand == "both" or strand == "plus"):
        app.Parameters['--strand'].on(strand)
    else:
        raise ValueError("Option --strand accepts only 'both'"
                         "or 'plus' values")
    app.Parameters['--derep_fulllength'].on(fasta_filepath)
    app.Parameters['--output'].on(output_filepath)
    app.Parameters['--log'].on(log_filepath)

    app_result = app()

    return output_filepath, uc_filepath, log_filepath