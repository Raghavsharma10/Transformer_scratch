def extract_from_fastq(fq):
    """Extract metrics from a fastq file.

    Return average quality and read length
    """
    for rec in SeqIO.parse(fq, "fastq"):
        yield nanomath.ave_qual(rec.letter_annotations["phred_quality"]), len(rec)