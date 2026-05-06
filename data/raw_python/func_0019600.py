def determine_file_type(fname):
    """
    Detect file type.

    The following file types are supported:
    BED, narrowPeak, FASTA, list of chr:start-end regions
    If the extension is bed, fa, fasta or narrowPeak, we will believe this
    without checking!

    Parameters
    ----------
    fname : str
        File name.

    Returns
    -------
    filetype : str
        Filename in lower-case.
    """
    if not (isinstance(fname, str) or isinstance(fname, unicode)):
        raise ValueError("{} is not a file name!", fname)

    if not os.path.isfile(fname):
        raise ValueError("{} is not a file!", fname)

    ext = os.path.splitext(fname)[1].lower()
    if ext in ["bed"]:
        return "bed"
    elif ext in ["fa", "fasta"]:
        return "fasta"
    elif ext in ["narrowpeak"]:
        return "narrowpeak"

    try:
        Fasta(fname)
        return "fasta"
    except:
        pass
    # Read first line that is not a comment or an UCSC-specific line
    p = re.compile(r'^(#|track|browser)') 
    with open(fname) as f:
        for line in f.readlines():
            line = line.strip()
            if not p.search(line):
                break
    region_p = re.compile(r'^(.+):(\d+)-(\d+)$')
    if region_p.search(line):
        return "region"
    else:
        vals = line.split("\t")
        if len(vals) >= 3:
            try:
                _, _ = int(vals[1]), int(vals[2])
            except ValueError:
                return "unknown"
            
            if len(vals) == 10:
                try: 
                    _, _ = int(vals[4]), int(vals[9])
                    return "narrowpeak"
                except ValueError:
                    # As far as I know there is no 10-column BED format
                    return "unknown"
                    pass
            return "bed"
    
    # Catch-all
    return "unknown"