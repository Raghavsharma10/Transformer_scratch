def usearch61_smallmem_cluster(intermediate_fasta,
                               percent_id=0.97,
                               minlen=64,
                               rev=False,
                               output_dir=".",
                               remove_usearch_logs=False,
                               wordlength=8,
                               usearch61_maxrejects=32,
                               usearch61_maxaccepts=1,
                               sizeorder=False,
                               HALT_EXEC=False,
                               output_uc_filepath=None,
                               log_name="smallmem_clustered.log",
                               sizeout=False,
                               consout_filepath=None):
    """ Performs usearch61 de novo clustering via cluster_smallmem option

    Only supposed to be used with length sorted data (and performs length
    sorting automatically) and does not support reverse strand matching

    intermediate_fasta:  fasta filepath to be clustered with usearch61
    percent_id:  percentage id to cluster at
    minlen: minimum sequence length
    rev: will enable reverse strand matching if True
    output_dir: directory to output log, OTU mapping, and intermediate files
    remove_usearch_logs: Saves usearch log files
    wordlength: word length to use for initial high probability sequence matches
    usearch61_maxrejects: Set to 'default' or an int value specifying max
     rejects
    usearch61_maxaccepts: Number of accepts allowed by usearch61
    HALT_EXEC: application controller option to halt execution
    output_uc_filepath: Path to write clusters (.uc) file.
    log_name: filepath to write usearch61 generated log file
    sizeout: If True, will save abundance data in output fasta labels.
    consout_filepath: Needs to be set to save clustered consensus fasta
     filepath used for chimera checking.
    """

    log_filepath = join(output_dir, log_name)

    params = {'--minseqlength': minlen,
              '--cluster_smallmem': intermediate_fasta,
              '--id': percent_id,
              '--uc': output_uc_filepath,
              '--wordlength': wordlength,
              '--maxrejects': usearch61_maxrejects,
              '--maxaccepts': usearch61_maxaccepts,
              '--usersort': True
              }

    if sizeorder:
        params['--sizeorder'] = True
    if not remove_usearch_logs:
        params['--log'] = log_filepath
    if rev:
        params['--strand'] = 'both'
    else:
        params['--strand'] = 'plus'
    if sizeout:
        params['--sizeout'] = True
    if consout_filepath:
        params['--consout'] = consout_filepath

    clusters_fp = output_uc_filepath

    app = Usearch61(params, WorkingDir=output_dir, HALT_EXEC=HALT_EXEC)

    app_result = app()

    return clusters_fp, app_result