def usearch_cluster_seqs_ref(
        fasta_filepath,
        output_filepath=None,
        percent_id=0.97,
        sizein=True,
        sizeout=True,
        w=64,
        slots=16769023,
        maxrejects=64,
        log_name="usearch_cluster_seqs.log",
        usersort=True,
        HALT_EXEC=False,
        save_intermediate_files=False,
        remove_usearch_logs=False,
        suppress_new_clusters=False,
        refseqs_fp=None,
        output_dir=None,
        working_dir=None,
        rev=False):
    """ Cluster seqs at percent_id, output consensus fasta

    Also appends de novo clustered seqs if suppress_new_clusters is False.
    Forced to handle reference + de novo in hackish fashion as usearch does not
    work as listed in the helpstrings.  Any failures are clustered de novo,
    and given unique cluster IDs.

    fasta_filepath = input fasta file, generally a dereplicated fasta
    output_filepath = output reference clustered uc filepath
    percent_id = minimum identity percent.
    sizein = not defined in usearch helpstring
    sizeout = not defined in usearch helpstring
    w = Word length for U-sorting
    slots = Size of compressed index table. Should be prime, e.g. 40000003.
     Should also specify --w, typical is --w 16 or --w 32.
    maxrejects = Max rejected targets, 0=ignore, default 32.
    log_name = string specifying output log name
    usersort = Enable if input fasta not sorted by length purposefully, lest
     usearch will raise an error.  In post chimera checked sequences, the seqs
     are sorted by abundance, so this should be set to True.
    HALT_EXEC: Used for debugging app controller
    save_intermediate_files: Preserve all intermediate files created.
    suppress_new_clusters: Disables de novo OTUs when ref based OTU picking
     enabled.
    refseqs_fp: Filepath for ref based OTU picking
    output_dir: output directory
    rev = search plus and minus strands of sequences
    """
    if not output_filepath:
        _, output_filepath = mkstemp(prefix='usearch_cluster_ref_based',
                                     suffix='.uc')

    log_filepath = join(working_dir, log_name)

    uc_filepath = join(working_dir, "clustered_seqs_post_chimera.uc")

    params = {'--sizein': sizein,
              '--sizeout': sizeout,
              '--id': percent_id,
              '--w': w,
              '--slots': slots,
              '--maxrejects': maxrejects}

    app = Usearch(params, WorkingDir=working_dir, HALT_EXEC=HALT_EXEC)

    if usersort:
        app.Parameters['--usersort'].on()
    if rev:
        app.Parameters['--rev'].on()

    data = {'--query': fasta_filepath,
            '--uc': uc_filepath,
            '--db': refseqs_fp
            }

    if not remove_usearch_logs:
        data['--log'] = log_filepath

    app_result = app(data)

    files_to_remove = []

    # Need to create fasta file of all hits (with reference IDs),
    # recluster failures if new clusters allowed, and create complete fasta
    # file, with unique fasta label IDs.

    if suppress_new_clusters:
        output_fna_filepath = join(output_dir, 'ref_clustered_seqs.fasta')
        output_filepath, labels_hits = get_fasta_from_uc_file(fasta_filepath,
                                                              uc_filepath, hit_type="H", output_dir=output_dir,
                                                              output_fna_filepath=output_fna_filepath)

        files_to_remove.append(uc_filepath)
    else:
        # Get fasta of successful ref based clusters
        output_fna_clustered = join(output_dir, 'ref_clustered_seqs.fasta')
        output_filepath_ref_clusters,  labels_hits =\
            get_fasta_from_uc_file(fasta_filepath, uc_filepath, hit_type="H",
                                   output_dir=output_dir, output_fna_filepath=output_fna_clustered)

        # get failures and recluster
        output_fna_failures =\
            join(output_dir, 'ref_clustered_seqs_failures.fasta')
        output_filepath_failures, labels_hits =\
            get_fasta_from_uc_file(fasta_filepath,
                                   uc_filepath, hit_type="N", output_dir=output_dir,
                                   output_fna_filepath=output_fna_failures)

        # de novo cluster the failures
        app_result, output_filepath_clustered_failures =\
            usearch_cluster_seqs(output_fna_failures, output_filepath=
                                 join(
                                     output_dir,
                                     'clustered_seqs_reference_failures.fasta'),
                                 percent_id=percent_id, sizein=sizein, sizeout=sizeout, w=w,
                                 slots=slots, maxrejects=maxrejects,
                                 save_intermediate_files=save_intermediate_files,
                                 remove_usearch_logs=remove_usearch_logs, working_dir=working_dir)

        output_filepath = concatenate_fastas(output_fna_clustered,
                                             output_fna_failures, output_concat_filepath=join(
                                                 output_dir,
                                                 'concatenated_reference_denovo_clusters.fasta'))

        files_to_remove.append(output_fna_clustered)
        files_to_remove.append(output_fna_failures)
        files_to_remove.append(output_filepath_clustered_failures)

    if not save_intermediate_files:
        remove_files(files_to_remove)

    return app_result, output_filepath