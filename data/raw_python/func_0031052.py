def combineReads(filename, sequences, readClass=DNARead,
                 upperCase=False, idPrefix='command-line-read-'):
    """
    Combine FASTA reads from a file and/or sequence strings.

    @param filename: A C{str} file name containing FASTA reads.
    @param sequences: A C{list} of C{str} sequences. If a sequence
        contains spaces, the last field (after splitting on spaces) will be
        used as the sequence and the first fields will be used as the sequence
        id.
    @param readClass: The class of the individual reads.
    @param upperCase: If C{True}, reads will be converted to upper case.
    @param idPrefix: The C{str} prefix that will be used for the id of the
        sequences in C{sequences} that do not have an id specified. A trailing
        sequence number will be appended to this prefix. Note that
        'command-line-read-', the default id prefix, could collide with ids in
        the FASTA file, if given. So output might be ambiguous. That's why we
        allow the caller to specify a custom prefix.
    @return: A C{FastaReads} instance.
    """
    # Read sequences from a FASTA file, if given.
    if filename:
        reads = FastaReads(filename, readClass=readClass, upperCase=upperCase)
    else:
        reads = Reads()

    # Add any individually specified subject sequences.
    if sequences:
        for count, sequence in enumerate(sequences, start=1):
            # Try splitting the sequence on its last space and using the
            # first part of the split as the read id. If there's no space,
            # assign a generic id.
            parts = sequence.rsplit(' ', 1)
            if len(parts) == 2:
                readId, sequence = parts
            else:
                readId = '%s%d' % (idPrefix, count)
            if upperCase:
                sequence = sequence.upper()
            read = readClass(readId, sequence)
            reads.add(read)

    return reads