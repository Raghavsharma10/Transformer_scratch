def parseFASTA(fastaFNH):
    """
    Parse the records in a FASTA-format file keeping the file open, and reading through
    one line at a time.

    :type source: path to FAST file or open file handle
    :param source: The data source from which to parse the FASTA records.
                   Expects the input to resolve to a collection that can be iterated
                   through, such as an open file handle.

    :rtype: tuple
    :return: FASTA records containing entries for id, description and data.
    """
    recs = []
    seq = []
    seqID = ""
    descr = ""

    for line in file_handle(fastaFNH):
        line = line.strip()
        if line[0] == ";":
            continue
        if line[0] == ">":
            # conclude previous record
            if seq:
                recs.append(FASTARecord(seqID, descr, "".join(seq)))
                seq = []
            # start new record
            line = line[1:].split(None, 1)
            seqID, descr = line[0], line[1]
        else:
            seq.append(line)

    # catch last seq in file
    if seq:
        recs.append(FASTARecord(seqID, descr, "".join(seq)))
    return recs