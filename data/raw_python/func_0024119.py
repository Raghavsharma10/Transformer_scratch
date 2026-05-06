def process_fasta(fasta, **kwargs):
    """Combine metrics extracted from a fasta file."""
    logging.info("Nanoget: Starting to collect statistics from a fasta file.")
    inputfasta = handle_compressed_input(fasta, file_type="fasta")
    return ut.reduce_memory_usage(pd.DataFrame(
        data=[len(rec) for rec in SeqIO.parse(inputfasta, "fasta")],
        columns=["lengths"]
    ).dropna())