def uclust_cluster_from_sorted_fasta_filepath(
        fasta_filepath,
        uc_save_filepath=None,
        percent_ID=0.97,
        max_accepts=1,
        max_rejects=8,
        stepwords=8,
        word_length=8,
        optimal=False,
        exact=False,
        suppress_sort=False,
        enable_rev_strand_matching=False,
        subject_fasta_filepath=None,
        suppress_new_clusters=False,
        stable_sort=False,
        tmp_dir=gettempdir(),
        HALT_EXEC=False):
    """ Returns clustered uclust file from sorted fasta"""
    output_filepath = uc_save_filepath
    if not output_filepath:
        _, output_filepath = mkstemp(dir=tmp_dir, prefix='uclust_clusters',
                                     suffix='.uc')

    params = {'--id': percent_ID,
              '--maxaccepts': max_accepts,
              '--maxrejects': max_rejects,
              '--stepwords': stepwords,
              '--w': word_length,
              '--tmpdir': tmp_dir}
    app = Uclust(params,
                 TmpDir=tmp_dir, HALT_EXEC=HALT_EXEC)

    # Set any additional parameters specified by the user
    if enable_rev_strand_matching:
        app.Parameters['--rev'].on()
    if optimal:
        app.Parameters['--optimal'].on()
    if exact:
        app.Parameters['--exact'].on()
    if suppress_sort:
        app.Parameters['--usersort'].on()
    if subject_fasta_filepath:
        app.Parameters['--lib'].on(subject_fasta_filepath)
    if suppress_new_clusters:
        app.Parameters['--libonly'].on()
    if stable_sort:
        app.Parameters['--stable_sort'].on()

    app_result = app({'--input': fasta_filepath, '--uc': output_filepath})
    return app_result