def create_bwa_index_from_fasta_file(fasta_in, params=None):
    """Create a BWA index from an input fasta file.

    fasta_in: the input fasta file from which to create the index
    params: dict of bwa index specific paramters

    This method returns a dictionary where the keys are the various
    output suffixes (.amb, .ann, .bwt, .pac, .sa) and the values
    are open file objects.

    The index prefix will be the same as fasta_in, unless the -p parameter
    is passed in params.
    """
    if params is None:
        params = {}

    # Instantiate the app controller
    index = BWA_index(params)

    # call the application, passing the fasta file in
    results = index({'fasta_in': fasta_in})
    return results