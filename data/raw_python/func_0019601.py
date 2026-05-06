def get_seqs_type(seqs):
    """
    automagically determine input type
    the following types are detected:
        - Fasta object
        - FASTA file
        - list of regions
        - region file
        - BED file
    """
    region_p = re.compile(r'^(.+):(\d+)-(\d+)$')
    if isinstance(seqs, Fasta):
        return "fasta"
    elif isinstance(seqs, list):
        if len(seqs) == 0:
            raise ValueError("empty list of sequences to scan")
        else:
            if region_p.search(seqs[0]):
                return "regions"
            else:
                raise ValueError("unknown region type")
    elif isinstance(seqs, str) or isinstance(seqs, unicode):
        if os.path.isfile(seqs):
            ftype = determine_file_type(seqs)
            if ftype == "unknown":
                raise ValueError("unknown type")
            elif ftype == "narrowpeak":
                raise ValueError("narrowPeak not yet supported in this function")
            else:
                return ftype + "file"
        else:
            raise ValueError("no file found with name {}".format(seqs))
    else:
        raise ValueError("unknown type {}".format(type(seqs).__name__))