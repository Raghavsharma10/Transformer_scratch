def assign_reads_to_otus(original_fasta,
                         filtered_fasta,
                         output_filepath=None,
                         log_name="assign_reads_to_otus.log",
                         perc_id_blast=0.97,
                         global_alignment=True,
                         HALT_EXEC=False,
                         save_intermediate_files=False,
                         remove_usearch_logs=False,
                         working_dir=None):
    """ Uses original fasta file, blasts to assign reads to filtered fasta

    original_fasta = filepath to original query fasta
    filtered_fasta = filepath to enumerated, filtered fasta
    output_filepath = output path to clusters (uc) file
    log_name = string specifying output log name
    perc_id_blast = percent ID for blasting original seqs against filtered set
    usersort = Enable if input fasta not sorted by length purposefully, lest
     usearch will raise an error.  In post chimera checked sequences, the seqs
     are sorted by abundance, so this should be set to True.
    HALT_EXEC: Used for debugging app controller
    save_intermediate_files: Preserve all intermediate files created.
    """

    # Not sure if I feel confortable using blast as a way to recapitulate
    # original read ids....
    if not output_filepath:
        _, output_filepath = mkstemp(prefix='assign_reads_to_otus',
                                     suffix='.uc')

    log_filepath = join(working_dir, log_name)

    params = {'--id': perc_id_blast,
              '--global': global_alignment}

    app = Usearch(params, WorkingDir=working_dir, HALT_EXEC=HALT_EXEC)

    data = {'--query': original_fasta,
            '--db': filtered_fasta,
            '--uc': output_filepath
            }

    if not remove_usearch_logs:
        data['--log'] = log_filepath

    app_result = app(data)

    return app_result, output_filepath