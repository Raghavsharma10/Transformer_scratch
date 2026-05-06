def assign_dna_reads_to_dna_database(query_fasta_fp, database_fasta_fp,
                                     output_fp, params=None):
    """Assign DNA reads to a database fasta of DNA sequences.

    Wraps assign_reads_to_database, setting database and query types. All
    parameters are set to default unless params is passed.

    query_fasta_fp: absolute path to the query fasta file containing DNA
                   sequences.
    database_fasta_fp: absolute path to the database fasta file containing
                      DNA sequences.
    output_fp: absolute path where the output file will be generated.
    params: optional. dict containing parameter settings to be used
                  instead of default values. Cannot change database or query
                  file types from dna and dna, respectively.

    This method returns an open file object. The output format
    defaults to blast9 and should be parsable by the PyCogent BLAST parsers.
    """
    if params is None:
        params = {}

    my_params = {'-t': 'dna',
                 '-q': 'dna'
                 }

    # if the user specified parameters other than default, then use them.
    # However, if they try to change the database or query types, raise an
    # applciation error.
    if '-t' in params or '-q' in params:
        raise ApplicationError("Cannot change database or query types when " +
                               "using assign_dna_reads_to_dna_database. " +
                               "Use assign_reads_to_database instead.\n")

    my_params.update(params)

    result = assign_reads_to_database(query_fasta_fp, database_fasta_fp,
                                      output_fp, my_params)

    return result