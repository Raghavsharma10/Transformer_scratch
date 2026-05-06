def get_fasta_record(fasta_file, transcript_name):
    """Return a single transcript from a valid fasta file as a record.

    record[transcript_name] = sequence

    Keyword arguments:
    fasta_file -- FASTA format file of the transcriptome
    transcript_name -- Name of the transcript as in the FASTA header

    """
    with open_pysam_file(fname=fasta_file, ftype='fasta') as f:
        sequence = f.fetch(transcript_name)
    return {transcript_name: sequence}