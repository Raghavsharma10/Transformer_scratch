def extract_all_from_fastq(rec):
    """Extract metrics from a fastq file.

    Return identifier, read length, average quality and median quality
    """
    return (rec.id,
            len(rec),
            nanomath.ave_qual(rec.letter_annotations["phred_quality"]),
            nanomath.median_qual(rec.letter_annotations["phred_quality"]))