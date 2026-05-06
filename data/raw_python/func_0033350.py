def uclust_search_and_align_from_fasta_filepath(
        query_fasta_filepath,
        subject_fasta_filepath,
        percent_ID=0.75,
        enable_rev_strand_matching=True,
        max_accepts=8,
        max_rejects=32,
        tmp_dir=gettempdir(),
        HALT_EXEC=False):
    """ query seqs against subject fasta using uclust,

       return global pw alignment of best match
    """

    # Explanation of parameter settings
    #  id - min percent id to count a match
    #  maxaccepts = 8 , searches for best match rather than first match
    #                   (0 => infinite accepts, or good matches before
    #                    quitting search)
    #  maxaccepts = 32,
    #  libonly = True , does not add sequences to the library if they don't
    #                   match something there already. this effectively makes
    #                   uclust a search tool rather than a clustering tool

    params = {'--id': percent_ID,
              '--maxaccepts': max_accepts,
              '--maxrejects': max_rejects,
              '--libonly': True,
              '--lib': subject_fasta_filepath,
              '--tmpdir': tmp_dir}

    if enable_rev_strand_matching:
        params['--rev'] = True

    # instantiate the application controller
    app = Uclust(params,
                 TmpDir=tmp_dir, HALT_EXEC=HALT_EXEC)

    # apply uclust
    _, alignment_filepath = mkstemp(dir=tmp_dir, prefix='uclust_alignments',
                                    suffix='.fasta')
    _, uc_filepath = mkstemp(dir=tmp_dir, prefix='uclust_results',
                             suffix='.uc')
    input_data = {'--input': query_fasta_filepath,
                  '--fastapairs': alignment_filepath,
                  '--uc': uc_filepath}
    app_result = app(input_data)

    # yield the pairwise alignments
    for result in process_uclust_pw_alignment_results(
            app_result['PairwiseAlignments'], app_result['ClusterFile']):
        try:
            yield result
        except GeneratorExit:
            break

    # clean up the temp files that were generated
    app_result.cleanUp()

    return