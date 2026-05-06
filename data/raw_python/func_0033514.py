def usearch61_chimera_check_denovo(abundance_fp,
                                   uchime_denovo_fp,
                                   minlen=64,
                                   output_dir=".",
                                   remove_usearch_logs=False,
                                   uchime_denovo_log_fp="uchime_denovo.log",
                                   usearch61_minh=0.28,
                                   usearch61_xn=8.0,
                                   usearch61_dn=1.4,
                                   usearch61_mindiffs=3,
                                   usearch61_mindiv=0.8,
                                   usearch61_abundance_skew=2.0,
                                   HALT_EXEC=False):
    """ Does de novo, abundance based chimera checking with usearch61

    abundance_fp: input consensus fasta file with abundance information for
     each cluster.
    uchime_denovo_fp: output uchime file for chimera results.
    minlen: minimum sequence length for usearch input fasta seqs.
    output_dir: output directory
    removed_usearch_logs: suppresses creation of log file.
    uchime_denovo_log_fp: output filepath for log file.
    usearch61_minh: Minimum score (h) to be classified as chimera.
     Increasing this value tends to the number of false positives (and also
     sensitivity).
    usearch61_xn:  Weight of "no" vote.  Increasing this value tends to the
     number of false positives (and also sensitivity).
    usearch61_dn:  Pseudo-count prior for "no" votes. (n). Increasing this
     value tends to the number of false positives (and also sensitivity).
    usearch61_mindiffs:  Minimum number of diffs in a segment. Increasing this
     value tends to reduce the number of false positives while reducing
     sensitivity to very low-divergence chimeras.
    usearch61_mindiv:  Minimum divergence, i.e. 100% - identity between the
     query and closest reference database sequence. Expressed as a percentage,
     so the default is 0.8%, which allows chimeras that are up to 99.2% similar
     to a reference sequence.
    usearch61_abundance_skew: abundance skew for de novo chimera comparisons.
    HALTEXEC: halt execution and returns command used for app controller.
    """

    params = {'--minseqlength': minlen,
              '--uchime_denovo': abundance_fp,
              '--uchimeout': uchime_denovo_fp,
              '--minh': usearch61_minh,
              '--xn': usearch61_xn,
              '--dn': usearch61_dn,
              '--mindiffs': usearch61_mindiffs,
              '--mindiv': usearch61_mindiv,
              '--abskew': usearch61_abundance_skew
              }

    if not remove_usearch_logs:
        params['--log'] = uchime_denovo_log_fp

    app = Usearch61(params, WorkingDir=output_dir, HALT_EXEC=HALT_EXEC)

    app_result = app()

    return uchime_denovo_fp, app_result