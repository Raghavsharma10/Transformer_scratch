def fetch_seq(ac, start_i=None, end_i=None):
    """Fetches sequences and subsequences from NCBI eutils and Ensembl
    REST interfaces.

    :param string ac: accession of sequence to fetch
    :param int start_i: start position of *interbase* interval
    :param int end_i: end position of *interbase* interval


    **IMPORTANT** start_i and end_i specify 0-based interbase
    coordinates, which refer to junctions between nucleotides.  This
    is numerically equivalent to 0-based, right-open nucleotide
    coordinates.

    Without an interval, the full sequence is returned::

    >> len(fetch_seq('NP_056374.2'))
    1596

    Therefore, it's preferable to provide the interval rather than
    using Python slicing sequence on the delivered sequence::

    >> fetch_seq('NP_056374.2',0,10)   # This!
    'MESRETLSSS'

    >> fetch_seq('NP_056374.2')[0:10]  # Not this!
    'MESRETLSSS'

    >> fetch_seq('NP_056374.2',0,10) == fetch_seq('NP_056374.2')[0:10]
    True

    Providing intervals is especially important for large sequences::

    >> fetch_seq('NC_000001.10',2000000,2000030)
    'ATCACACGTGCAGGAACCCTTTTCCAAAGG'

    This call will pull back 30 bases plus overhead; without the
    interval, one would receive 250MB of chr1 plus overhead!

    Essentially any RefSeq, Genbank, BIC, or Ensembl sequence may be
    fetched:

    >> [(ac,fetch_seq(ac,0,25))
    ... for ac in ['NG_032072.1', 'NW_003571030.1', 'NT_113901.1',
    ... 'NC_000001.10','NP_056374.2', 'GL000191.1', 'KB663603.1',
    ... 'ENST00000288602', 'ENSP00000288602']] # doctest: +NORMALIZE_WHITESPACE
    [('NG_032072.1', 'AAAATTAAATTAAAATAAATAAAAA'),
     ('NW_003571030.1', 'TTGTGTGTTAGGGTGCTCTAAGCAA'),
     ('NT_113901.1', 'GAATTCCTCGTTCACACAGTTTCTT'),
     ('NC_000001.10', 'NNNNNNNNNNNNNNNNNNNNNNNNN'),
     ('NP_056374.2', 'MESRETLSSSRQRGGESDFLPVSSA'),
     ('GL000191.1', 'GATCCACCTGCCTCAGCCTCCCAGA'),
     ('KB663603.1', 'TTTATTTATTTTAGATACTTATCTC'),
     ('ENST00000288602', u'CGCCTCCCTTCCCCCTCCCCGCCCG'),
     ('ENSP00000288602', u'MAALSGGGGGGAEPGQALFNGDMEP')]


    RuntimeError is thrown in the case of errors::

    >> fetch_seq('NM_9.9')
    Traceback (most recent call last):
       ...
    RuntimeError: No sequence available for NM_9.9

    >> fetch_seq('QQ01234')
    Traceback (most recent call last):
       ...
    RuntimeError: No sequence fetcher for QQ01234

    """

    ac_dispatch = [
        {
            "re": re.compile(r"^(?:AC|N[CGMPRTW])_|^[A-L]\w\d|^U\d"),
            "fetcher": _fetch_seq_ncbi
        },
        {
            "re": re.compile(r"^ENS[TP]\d+"),
            "fetcher": _fetch_seq_ensembl
        },
    ]

    eligible_fetchers = [
        dr["fetcher"] for dr in ac_dispatch if dr["re"].match(ac)
    ]

    if len(eligible_fetchers) == 0:
        raise RuntimeError("No sequence fetcher for {ac}".format(ac=ac))

    if len(eligible_fetchers) >= 1:  # pragma: nocover (no way to test)
        _logger.debug("Multiple sequence fetchers found for "
                     "{ac}; using first".format(ac=ac))

    fetcher = eligible_fetchers[0]
    _logger.debug("fetching {ac} with {f}".format(ac=ac, f=fetcher))

    try:
        return fetcher(ac, start_i, end_i)
    except requests.RequestException as ex:
        raise RuntimeError("Failed to fetch {ac} ({ex})".format(ac=ac, ex=ex))