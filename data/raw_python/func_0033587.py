def assign_dna_reads_to_dna_database(query_fasta_fp, database_fasta_fp, out_fp,
                                     params={}):
    """Wraps assign_reads_to_database, setting various parameters.

    The default settings are below, but may be overwritten and/or added to
    using the params dict:

    algorithm:      bwasw
    """
    my_params = {'algorithm': 'bwasw'}
    my_params.update(params)

    result = assign_reads_to_database(query_fasta_fp, database_fasta_fp,
                                      out_fp, my_params)

    return result