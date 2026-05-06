def sort_by_length_usearch61(seq_path,
                             output_dir=".",
                             minlen=64,
                             remove_usearch_logs=False,
                             HALT_EXEC=False,
                             output_fna_filepath=None,
                             log_name="length_sorted.log"):
    """ usearch61 application call to sort fasta file by length.

    seq_path:  fasta filepath to be clustered with usearch61
    output_dir: directory to output log, OTU mapping, and intermediate files
    minlen: minimum sequence length
    remove_usearch_logs: Saves usearch log files
    HALT_EXEC: application controller option to halt execution
    output_fna_filepath: path to write sorted fasta filepath
    log_name: filepath to write usearch61 generated log file
    """

    if not output_fna_filepath:
        _, output_fna_filepath = mkstemp(prefix='length_sorted', suffix='.fna')

    log_filepath = join(output_dir, log_name)

    params = {'--minseqlength': minlen,
              '--sortbylength': seq_path,
              '--output': output_fna_filepath
              }
    if not remove_usearch_logs:
        params['--log'] = log_filepath

    app = Usearch61(params, WorkingDir=output_dir, HALT_EXEC=HALT_EXEC)

    app_result = app()

    return output_fna_filepath, app_result