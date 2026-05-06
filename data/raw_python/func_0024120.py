def process_fastq_plain(fastq, **kwargs):
    """Combine metrics extracted from a fastq file."""
    logging.info("Nanoget: Starting to collect statistics from plain fastq file.")
    inputfastq = handle_compressed_input(fastq)
    return ut.reduce_memory_usage(pd.DataFrame(
        data=[res for res in extract_from_fastq(inputfastq) if res],
        columns=["quals", "lengths"]
    ).dropna())