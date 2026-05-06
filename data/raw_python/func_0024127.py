def process_fastq_minimal(fastq, **kwargs):
    """Swiftly extract minimal features (length and timestamp) from a rich fastq file"""
    infastq = handle_compressed_input(fastq)
    try:
        df = pd.DataFrame(
            data=[rec for rec in fq_minimal(infastq) if rec],
            columns=["timestamp", "lengths"]
        )
    except IndexError:
        logging.error("Fatal: Incorrect file structure for fastq_minimal")
        sys.exit("Error: file does not match expected structure for fastq_minimal")
    return ut.reduce_memory_usage(df)