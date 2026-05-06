def sortmerna_map(seq_path,
                  output_dir,
                  refseqs_fp,
                  sortmerna_db,
                  e_value=1,
                  threads=1,
                  best=None,
                  num_alignments=None,
                  HALT_EXEC=False,
                  output_sam=False,
                  sam_SQ_tags=False,
                  blast_format=3,
                  print_all_reads=True,
                  ):
    """Launch sortmerna mapper

        Parameters
        ----------
        seq_path : str
            filepath to reads.
        output_dir : str
            dirpath to sortmerna output.
        refseqs_fp : str
            filepath of reference sequences.
        sortmerna_db : str
            indexed reference database.
        e_value : float, optional
            E-value threshold [default: 1].
        threads : int, optional
            number of threads to use (OpenMP) [default: 1].
        best : int, optional
            number of best alignments to output per read
            [default: None].
        num_alignments : int, optional
            number of first alignments passing E-value threshold to
            output per read [default: None].
        HALT_EXEC : bool, debugging parameter
            If passed, will exit just before the sortmerna command
            is issued and will print out the command that would
            have been called to stdout [default: False].
        output_sam : bool, optional
            flag to set SAM output format [default: False].
        sam_SQ_tags : bool, optional
            add SQ field to SAM output (if output_SAM is True)
            [default: False].
        blast_format : int, optional
            Output Blast m8 tabular + 2 extra columns for CIGAR
            string and query coverge [default: 3].
        print_all_reads : bool, optional
            output NULL alignments for non-aligned reads
            [default: True].

        Returns
        -------
        dict of result paths set in _get_result_paths()
    """

    if not (blast_format or output_sam):
        raise ValueError("Either Blast or SAM output alignment "
                         "format must be chosen.")

    if (best and num_alignments):
        raise ValueError("Only one of --best or --num_alignments "
                         "options must be chosen.")

    # Instantiate the object
    smr = Sortmerna(HALT_EXEC=HALT_EXEC)

    # Set the input reference sequence + indexed database path
    smr.Parameters['--ref'].on("%s,%s" % (refseqs_fp, sortmerna_db))

    # Set input query sequences path
    smr.Parameters['--reads'].on(seq_path)

    # Set Blast tabular output
    # The option --blast 3 represents an
    # m8 blast tabular output + two extra
    # columns containing the CIGAR string
    # and the query coverage
    if blast_format:
        smr.Parameters['--blast'].on(blast_format)

    # Output alignments in SAM format
    if output_sam:
        smr.Parameters['--sam'].on()
        if sam_SQ_tags:
            smr.Parameters['--SQ'].on()

    # Turn on NULL string alignment output
    if print_all_reads:
        smr.Parameters['--print_all_reads'].on()

    # Set output results path (for Blast alignments and log file)
    output_file = join(output_dir, "sortmerna_map")
    smr.Parameters['--aligned'].on(output_file)

    # Set E-value threshold
    if e_value is not None:
        smr.Parameters['-e'].on(e_value)

    # Set number of best alignments to output per read
    if best is not None:
        smr.Parameters['--best'].on(best)

    # Set number of first alignments passing E-value threshold
    # to output per read
    if num_alignments is not None:
        smr.Parameters['--num_alignments'].on(num_alignments)

    # Set number of threads
    if threads is not None:
        smr.Parameters['-a'].on(threads)

    # Turn off parameters related to OTU-picking
    smr.Parameters['--fastx'].off()
    smr.Parameters['--otu_map'].off()
    smr.Parameters['--de_novo_otu'].off()
    smr.Parameters['--id'].off()
    smr.Parameters['--coverage'].off()

    # Run sortmerna
    app_result = smr()

    return app_result