def find_files(sequencepath):
    """
    Use glob to find all FASTA files in the provided sequence path. NOTE: FASTA files must have an extension such as
    .fasta, .fa, or .fas. Extensions of .fsa, .tfa, ect. are not currently supported
    :param sequencepath: path of folder containing FASTA genomes
    :return: list of FASTA files
    """
    # Create a sorted list of all the FASTA files in the sequence path
    files = sorted(glob(os.path.join(sequencepath, '*.fa*')))
    return files