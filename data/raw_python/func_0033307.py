def assign_reads_to_database(query_fasta_fp, database_fasta_fp, output_fp,
                             params=None):
    """Assign a set of query sequences to a reference database

    query_fasta_fp : absolute file path to query sequences
    database_fasta_fp : absolute file path to the reference database
    output_fp : absolute file path of the output file to write
    params : dict of BLAT specific parameters.

    This method returns an open file object. The output format
    defaults to blast9 and should be parsable by the PyCogent BLAST parsers.
    """
    if params is None:
        params = {}
    if '-out' not in params:
        params['-out'] = 'blast9'
    blat = Blat(params=params)

    result = blat([query_fasta_fp, database_fasta_fp, output_fp])
    return result['output']