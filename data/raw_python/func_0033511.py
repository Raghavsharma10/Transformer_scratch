def usearch61_cluster_ref(intermediate_fasta,
                          refseqs_fp,
                          percent_id=0.97,
                          rev=False,
                          minlen=64,
                          output_dir=".",
                          remove_usearch_logs=False,
                          wordlength=8,
                          usearch61_maxrejects=32,
                          usearch61_maxaccepts=1,
                          HALT_EXEC=False,
                          output_uc_filepath=None,
                          log_filepath="ref_clustered.log",
                          threads=1.0
                          ):
    """ Cluster input fasta seqs against reference database

    seq_path:  fasta filepath to be clustered with usearch61
    refseqs_fp: reference fasta filepath, used to cluster sequences against.
    percent_id:  percentage id to cluster at
    rev: enable reverse strand matching for clustering
    minlen: minimum sequence length
    output_dir: directory to output log, OTU mapping, and intermediate files
    remove_usearch_logs: Saves usearch log files
    wordlength: word length to use for clustering
    usearch61_maxrejects: Number of rejects allowed by usearch61
    usearch61_maxaccepts: Number of accepts allowed by usearch61
    output_uc_filepath: path to write usearch61 generated .uc file
    threads: Specify number of threads used per core per CPU
    HALT_EXEC: application controller option to halt execution.
    """

    log_filepath = join(output_dir, log_filepath)

    params = {
        '--usearch_global': intermediate_fasta,
        '--db': refseqs_fp,
        '--minseqlength': minlen,
        '--id': percent_id,
        '--uc': output_uc_filepath,
        '--wordlength': wordlength,
        '--maxrejects': usearch61_maxrejects,
        '--maxaccepts': usearch61_maxaccepts,
        '--threads': threads
    }

    if not remove_usearch_logs:
        params['--log'] = log_filepath
    if rev:
        params['--strand'] = 'both'
    else:
        params['--strand'] = 'plus'

    clusters_fp = output_uc_filepath

    app = Usearch61(params, WorkingDir=output_dir, HALT_EXEC=HALT_EXEC)

    app_result = app()

    return clusters_fp, app_result