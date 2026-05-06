def parse_map_file(mapFNH):
    """
    Opens a QIIME mapping file and stores the contents in a dictionary keyed on SampleID
    (default) or a user-supplied one. The only required fields are SampleID,
    BarcodeSequence, LinkerPrimerSequence (in that order), and Description
    (which must be the final field).

    :type mapFNH: str
    :param mapFNH: Either the full path to the map file or an open file handle

    :rtype: tuple, dict
    :return: A tuple of header line for mapping file and a map associating each line of
             the mapping file with the appropriate sample ID (each value of the map also
             contains the sample ID). An OrderedDict is used for mapping so the returned
             map is guaranteed to have the same order as the input file.

    Example data:
    #SampleID BarcodeSequence LinkerPrimerSequence State   Description
    11.V13    ACGCTCGACA      GTTTGATCCTGGCTCAG    Disease Rat_Oral
    """
    m = OrderedDict()
    map_header = None

    with file_handle(mapFNH) as mapF:
        for line in mapF:
            if line.startswith("#SampleID"):
                map_header = line.strip().split("\t")
            if line.startswith("#") or not line:
                    continue
            line = line.strip().split("\t")
            m[line[0]] = line

    return map_header, m