def fq_minimal(fq):
    """Minimal fastq metrics extractor.

    Quickly parse a fasta/fastq file - but makes expectations on the file format
    There will be dragons if unexpected format is used
    Expects a fastq_rich format, but extracts only timestamp and length
    """
    try:
        while True:
            time = next(fq)[1:].split(" ")[4][11:-1]
            length = len(next(fq))
            next(fq)
            next(fq)
            yield time, length
    except StopIteration:
        yield None