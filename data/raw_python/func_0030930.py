def check(fastaFile, jsonFiles):
    """
    Check for simple consistency between the FASTA file and the JSON files.

    Note that some checking is already performed by the BlastReadsAlignments
    class. That includes checking the number of reads matches the number of
    BLAST records and that read ids and BLAST record read ids match.

    @param jsonFiles: A C{list} of names of our BLAST JSON. These may
        may be compressed (as bz2).
    @param fastaFile: The C{str} name of a FASTA-containing file.
    """
    reads = FastaReads(fastaFile)
    readsAlignments = BlastReadsAlignments(reads, jsonFiles)
    for index, readAlignments in enumerate(readsAlignments):

        # Check that all the alignments in the BLAST JSON do not have query
        # sequences or query offsets that are greater than the length of
        # the sequence given in the FASTA file.
        fastaLen = len(readAlignments.read)
        for readAlignment in readAlignments:
            for hsp in readAlignment.hsps:
                # The FASTA sequence should be at least as long as the
                # query in the JSON BLAST record (minus any gaps).
                assert (fastaLen >=
                        len(hsp.query) - hsp.query.count('-')), (
                    'record %d: FASTA len %d < HSP query len %d.\n'
                    'FASTA: %s\nQuery match: %s' % (
                        index, fastaLen, len(hsp.query),
                        readAlignments.read.sequence, hsp.query))
                # The FASTA sequence length should be larger than either of
                # the query offsets mentioned in the JSON BLAST
                # record. That's because readStart and readEnd are offsets
                # into the read - so they can't be bigger than the read
                # length.
                #
                # TODO: These asserts should be more informative when they
                # fail.
                assert fastaLen >= hsp.readEnd >= hsp.readStart, (
                    'record %d: FASTA len %d not greater than both read '
                    'offsets (%d - %d), or read offsets are non-increasing. '
                    'FASTA: %s\nQuery match: %s' % (
                        index, fastaLen, hsp.readStart, hsp.readEnd,
                        readAlignments.read.sequence, hsp.query))