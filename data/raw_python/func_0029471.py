def storeFASTA(fastaFNH):
    """
    Parse the records in a FASTA-format file by first reading the entire file into memory.

    :type source: path to FAST file or open file handle
    :param source: The data source from which to parse the FASTA records. Expects the
                   input to resolve to a collection that can be iterated through, such as
                   an open file handle.

    :rtype: tuple
    :return: FASTA records containing entries for id, description and data.
    """
    fasta = file_handle(fastaFNH).read()
    return [FASTARecord(rec[0].split()[0], rec[0].split(None, 1)[1], "".join(rec[1:]))
            for rec in (x.strip().split("\n") for x in fasta.split(">")[1:])]