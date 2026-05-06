def get_first_transcript_name(fasta_file):
    """Return the first FASTA sequence from the given FASTA file.

    Keyword arguments:
    fasta_file -- FASTA format file of the transcriptome

    """
    with open_pysam_file(fname=fasta_file, ftype='fasta') as f:
        transcript_name = f.references[0]
    return transcript_name