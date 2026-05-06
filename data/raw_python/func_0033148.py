def sortmerna_ref_cluster(seq_path=None,
                          sortmerna_db=None,
                          refseqs_fp=None,
                          result_path=None,
                          tabular=False,
                          max_e_value=1,
                          similarity=0.97,
                          coverage=0.97,
                          threads=1,
                          best=1,
                          HALT_EXEC=False
                          ):
    """Launch sortmerna OTU picker

        Parameters
        ----------
        seq_path : str
            filepath to query sequences.
        sortmerna_db : str
            indexed reference database.
        refseqs_fp : str
            filepath of reference sequences.
        result_path : str
            filepath to output OTU map.
        max_e_value : float, optional
            E-value threshold [default: 1].
        similarity : float, optional
            similarity %id threshold [default: 0.97].
        coverage : float, optional
            query coverage % threshold [default: 0.97].
        threads : int, optional
            number of threads to use (OpenMP) [default: 1].
        tabular : bool, optional
            output BLAST tabular alignments [default: False].
        best : int, optional
            number of best alignments to output per read
            [default: 1].

        Returns
        -------
        clusters : dict of lists
            OTU ids and reads mapping to them

        failures : list
            reads which did not align
    """

    # Instantiate the object
    smr = Sortmerna(HALT_EXEC=HALT_EXEC)

    # Set input query sequences path
    if seq_path is not None:
        smr.Parameters['--reads'].on(seq_path)
    else:
        raise ValueError("Error: a read file is mandatory input.")

    # Set the input reference sequence + indexed database path
    if sortmerna_db is not None:
        smr.Parameters['--ref'].on("%s,%s" % (refseqs_fp, sortmerna_db))
    else:
        raise ValueError("Error: an indexed database for reference set %s must"
                         " already exist.\nUse indexdb_rna to index the"
                         " database." % refseqs_fp)

    if result_path is None:
        raise ValueError("Error: the result path must be set.")

    # Set output results path (for Blast alignments, clusters and failures)
    output_dir = dirname(result_path)
    if output_dir is not None:
        output_file = join(output_dir, "sortmerna_otus")
        smr.Parameters['--aligned'].on(output_file)

    # Set E-value threshold
    if max_e_value is not None:
        smr.Parameters['-e'].on(max_e_value)

    # Set similarity threshold
    if similarity is not None:
        smr.Parameters['--id'].on(similarity)

    # Set query coverage threshold
    if coverage is not None:
        smr.Parameters['--coverage'].on(coverage)

    # Set number of best alignments to output
    if best is not None:
        smr.Parameters['--best'].on(best)

    # Set Blast tabular output
    # The option --blast 3 represents an
    # m8 blast tabular output + two extra
    # columns containing the CIGAR string
    # and the query coverage
    if tabular:
        smr.Parameters['--blast'].on("3")

    # Set number of threads
    if threads is not None:
        smr.Parameters['-a'].on(threads)

    # Run sortmerna
    app_result = smr()

    # Put clusters into a map of lists
    f_otumap = app_result['OtuMap']
    rows = (line.strip().split('\t') for line in f_otumap)
    clusters = {r[0]: r[1:] for r in rows}

    # Put failures into a list
    f_failure = app_result['FastaForDenovo']
    failures = [re.split('>| ', label)[0]
                for label, seq in parse_fasta(f_failure)]

    # remove the aligned FASTA file and failures FASTA file
    # (currently these are re-constructed using pick_rep_set.py
    # further in the OTU-picking pipeline)
    smr_files_to_remove = [app_result['FastaForDenovo'].name,
                           app_result['FastaMatches'].name,
                           app_result['OtuMap'].name]

    return clusters, failures, smr_files_to_remove