def sort_by_abundance_usearch61(seq_path,
                                output_dir='.',
                                rev=False,
                                minlen=64,
                                remove_usearch_logs=False,
                                HALT_EXEC=False,
                                output_fna_filepath=None,
                                output_uc_filepath=None,
                                log_name="abundance_sorted.log",
                                threads=1.0):
    """ usearch61 application call to sort fasta file by abundance.

    seq_path:  fasta filepath to be clustered with usearch61
    output_dir: directory to output log, OTU mapping, and intermediate files
    rev: enable reverse strand matching for clustering/sorting
    minlen: minimum sequence length
    remove_usearch_logs: Saves usearch log files
    HALT_EXEC: application controller option to halt execution
    output_fna_filepath: path to write sorted fasta filepath
    output_uc_filepath: path to write usearch61 generated .uc file
    log_name: filepath to write usearch61 generated log file
    threads: Specify number of threads used per core per CPU
    """

    if not output_fna_filepath:
        _, output_fna_filepath = mkstemp(prefix='abundance_sorted',
                                         suffix='.fna')

    if not output_uc_filepath:
        _, output_uc_filepath = mkstemp(prefix='abundance_sorted',
                                        suffix='.uc')

    log_filepath = join(output_dir, log_name)

    params = {'--minseqlength': minlen,
              '--sizeout': True,
              '--derep_fulllength': seq_path,
              '--output': output_fna_filepath,
              '--uc': output_uc_filepath,
              '--threads': threads
              }

    if rev:
        params['--strand'] = 'both'
    if not remove_usearch_logs:
        params['--log'] = log_filepath

    app = Usearch61(params, WorkingDir=output_dir, HALT_EXEC=HALT_EXEC)

    app_result = app()

    return output_fna_filepath, output_uc_filepath, app_result