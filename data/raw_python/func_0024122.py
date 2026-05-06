def stream_fastq_full(fastq, threads):
    """Generator for returning metrics extracted from fastq.

    Extract from a fastq file:
    -readname
    -average and median quality
    -read_lenght
    """
    logging.info("Nanoget: Starting to collect full metrics from plain fastq file.")
    inputfastq = handle_compressed_input(fastq)
    with cfutures.ProcessPoolExecutor(max_workers=threads) as executor:
        for results in executor.map(extract_all_from_fastq, SeqIO.parse(inputfastq, "fastq")):
            yield results
    logging.info("Nanoget: Finished collecting statistics from plain fastq file.")