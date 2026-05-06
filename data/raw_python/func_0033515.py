def usearch61_chimera_check_ref(abundance_fp,
                                uchime_ref_fp,
                                reference_seqs_fp,
                                minlen=64,
                                output_dir=".",
                                remove_usearch_logs=False,
                                uchime_ref_log_fp="uchime_ref.log",
                                usearch61_minh=0.28,
                                usearch61_xn=8.0,
                                usearch61_dn=1.4,
                                usearch61_mindiffs=3,
                                usearch61_mindiv=0.8,
                                threads=1.0,
                                HALT_EXEC=False):
    """ Does reference based chimera checking with usearch61

    abundance_fp: input consensus fasta file with abundance information for
     each cluster.
    uchime_ref_fp: output uchime filepath for reference results
    reference_seqs_fp: reference fasta database for chimera checking.
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
    threads: Specify number of threads used per core per CPU
    HALTEXEC: halt execution and returns command used for app controller.
    """

    params = {'--minseqlength': minlen,
              '--uchime_ref': abundance_fp,
              '--uchimeout': uchime_ref_fp,
              '--db': reference_seqs_fp,
              '--minh': usearch61_minh,
              '--xn': usearch61_xn,
              '--dn': usearch61_dn,
              '--mindiffs': usearch61_mindiffs,
              '--mindiv': usearch61_mindiv,
              # Only works in plus according to usearch doc
              '--strand': 'plus',
              '--threads': threads
              }

    if not remove_usearch_logs:
        params['--log'] = uchime_ref_log_fp

    app = Usearch61(params, WorkingDir=output_dir, HALT_EXEC=HALT_EXEC)

    app_result = app()

    return uchime_ref_fp, app_result