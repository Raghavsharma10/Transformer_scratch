def is_fasta_valid(fasta_file):
    """Check if fasta file is valid. Raises a ValueError if pysam cannot read the file.

    #TODO: pysam  does not differentiate between BAM and SAM
    """
    try:
        f = pysam.FastaFile(fasta_file)
    except IOError:
        raise
    else:
        f.close()
    return True