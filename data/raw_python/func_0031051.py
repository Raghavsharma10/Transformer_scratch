def fastaSubtract(fastaFiles):
    """
    Given a list of open file descriptors, each with FASTA content,
    remove the reads found in the 2nd, 3rd, etc files from the first file
    in the list.

    @param fastaFiles: a C{list} of FASTA filenames.
    @raises IndexError: if passed an empty list.
    @return: An iterator producing C{Bio.SeqRecord} instances suitable for
        writing to a file using C{Bio.SeqIO.write}.

    """
    reads = {}
    firstFile = fastaFiles.pop(0)
    for seq in SeqIO.parse(firstFile, 'fasta'):
        reads[seq.id] = seq

    for fastaFile in fastaFiles:
        for seq in SeqIO.parse(fastaFile, 'fasta'):
            # Make sure that reads with the same id have the same sequence.
            if seq.id in reads:
                assert str(seq.seq) == str(reads[seq.id].seq)
            reads.pop(seq.id, None)

    return iter(reads.values())