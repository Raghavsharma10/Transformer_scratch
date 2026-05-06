def get_sample_data(sample_file):
    """Read and returns sample data to fill form with default sample sequence.  """

    sequence_sample_in_fasta = None
    with open(sample_file) as handle:
        sequence_sample_in_fasta = handle.read()

    return sequence_sample_in_fasta