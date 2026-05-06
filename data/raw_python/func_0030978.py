def read_gene2acc(file_path, logger):
    """Extracts Entrez ID -> gene symbol mapping from gene2accession.gz file.

    Parameters
    ----------
    file_path: str
        The path of the gene2accession.gz file (or a filtered version thereof).
        The file may be gzip'ed.

    Returns
    -------
    dict
        A mapping of Entrez IDs to gene symbols.
    """
    gene2acc = {}
    with misc.smart_open_read(file_path, mode='rb', try_gzip=True) as fh:
        reader = csv.reader(fh, dialect='excel-tab')
        next(reader)  # skip header
        for i, l in enumerate(reader):
            id_ = int(l[1])
            symbol = l[15]

            try:
                gene2acc[id_].append(symbol)
            except KeyError:
                gene2acc[id_] = [symbol]

            # print (l[0],l[15])

    # make sure all EntrezIDs map to a unique gene symbol
    n = len(gene2acc)
    for k, v in gene2acc.items():
        symbols = sorted(set(v))
        assert len(symbols) == 1
        gene2acc[k] = symbols[0]

    all_symbols = sorted(set(gene2acc.values()))
    m = len(all_symbols)

    logger.info('Found %d Entrez Gene IDs associated with %d gene symbols.',
                n, m)
    return gene2acc