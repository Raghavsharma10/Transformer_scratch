def parse_gaf(path_or_buffer, gene_ontology, valid_genes=None,
              db=None, ev_codes=None):
    """Parse a GAF 2.1 file containing GO annotations.
    
    Parameters
    ----------
    path_or_buffer : str or buffer
        The GAF file.
    gene_ontology : `GeneOntology`
        The Gene Ontology.
    valid_genes : Iterable of str, optional
        A list of valid gene names. [None]
    db : str, optional
        Select only annotations with this "DB"" value. [None]
    ev_codes : str or set of str, optional
        Select only annotations with this/these evidence codes. [None]
    
    Returns
    -------
    list of `GOAnnotation`
        The list of GO annotations.
    """
    #if path == '-':
    #    path = sys.stdin

    assert isinstance(gene_ontology, GeneOntology)
    if db is not None:
        assert isinstance(db, (str, _oldstr))
    if (ev_codes is not None) and ev_codes:
        assert isinstance(ev_codes, (str, _oldstr)) or \
                isinstance(ev_codes, Iterable)

    if isinstance(ev_codes, str):
        ev_codes = set([ev_codes])
    elif (ev_codes is not None) and ev_codes:
        ev_codes = set(ev_codes)
    else:
        ev_codes = None

    # open file, if necessary
    if isinstance(path_or_buffer, (str, _oldstr)):
        buffer = misc.gzip_open_text(path_or_buffer, encoding='ascii')
    else:
        buffer = path_or_buffer

    if valid_genes is not None:
        valid_genes = set(valid_genes)

    # use pandas to parse the file quickly
    df = pd.read_csv(buffer, sep='\t', comment='!', header=None, dtype=_oldstr)

    # replace pandas' NaNs with empty strings
    df.fillna('', inplace=True)

    # exclude annotations with unknown Gene Ontology terms
    all_go_term_ids = set(gene_ontology._term_dict.keys())
    sel = df.iloc[:, 4].isin(all_go_term_ids)
    logger.info(
        'Ignoring %d / %d annotations (%.1f %%) with unknown GO terms.',
        (~sel).sum(), sel.size, 100*((~sel).sum()/float(sel.size)))
    df = df.loc[sel]

    # filter rows for valid genes
    if valid_genes is not None:
        sel = df.iloc[:, 2].isin(valid_genes)
        logger.info(
            'Ignoring %d / %d annotations (%.1f %%) with unknown genes.',
            (~sel).sum(), sel.size, 100*((~sel).sum()/float(sel.size)))
        df = df.loc[sel]

    # filter rows for DB value
    if db is not None:
        sel = (df.iloc[:, 0] == db)
        logger.info(
            'Excluding %d / %d annotations (%.1f %%) with wrong DB values.',
            (~sel).sum(), sel.size, 100*((~sel).sum()/float(sel.size)))
        df = df.loc[sel]

    # filter rows for evidence value
    if ev_codes is not None:
        sel = (df.iloc[:, 6].isin(ev_codes))
        logger.info(
            'Excluding %d / %d annotations (%.1f %%) based on evidence code.',
            (~sel).sum(), sel.size, 100*((~sel).sum()/float(sel.size)))
        df = df.loc[sel]

    # convert each row into a GOAnnotation object
    go_annotations = []
    for i, l in df.iterrows():
        ann = GOAnnotation.from_list(gene_ontology, l.tolist())
        go_annotations.append(ann)
    logger.info('Read %d GO annotations.', len(go_annotations))

    return go_annotations