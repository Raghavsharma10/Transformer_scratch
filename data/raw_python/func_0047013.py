def get_fasta_records(fasta, transcripts):
    """Return list of transcript records from the given fasta file.
    Each record will be of the form {'sequence_id': {'sequence': 'AAA', 'length': 3}}

    trascripts should be provided as a list of sequence id's.

    """
    records = {}
    f = pysam.FastaFile(fasta)
    for transcript in transcripts:
        try:
            sequence, length = f.fetch(transcript), f.get_reference_length(transcript)
        except KeyError:
            msg = 'Transcript "{}" does not exist in transcriptome FASTA file'.format(transcript)
            log.error(msg)
            raise ArgumentError(msg)
        records[transcript] = {'sequence': sequence, 'length': length}
    f.close()
    return records