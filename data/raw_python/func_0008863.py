def load_catalog(filename):
    """
    Load a catalogue and extract the source positions (only)

    Parameters
    ----------
    filename : str
        Filename to read. Supported types are csv, tab, tex, vo, vot, and xml.

    Returns
    -------
    catalogue : list
        A list of [ (ra, dec), ...]

    """
    supported = get_table_formats()

    fmt = os.path.splitext(filename)[-1][1:].lower()  # extension sans '.'

    if fmt in ['csv', 'tab', 'tex'] and fmt in supported:
        log.info("Reading file {0}".format(filename))
        t = ascii.read(filename)
        catalog = list(zip(t.columns['ra'], t.columns['dec']))

    elif fmt in ['vo', 'vot', 'xml'] and fmt in supported:
        log.info("Reading file {0}".format(filename))
        t = parse_single_table(filename)
        catalog = list(zip(t.array['ra'].tolist(), t.array['dec'].tolist()))

    else:
        log.info("Assuming ascii format, reading first two columns")
        lines = [a.strip().split() for a in open(filename, 'r').readlines() if not a.startswith('#')]
        try:
            catalog = [(float(a[0]), float(a[1])) for a in lines]
        except:
            log.error("Expecting two columns of floats but failed to parse")
            log.error("Catalog file {0} not loaded".format(filename))
            raise Exception("Could not determine file format")

    return catalog