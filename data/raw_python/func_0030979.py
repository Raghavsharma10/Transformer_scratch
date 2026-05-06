def write_entrez2gene(file_path, entrez2gene, logger):
    """Writes Entrez ID -> gene symbol mapping to a tab-delimited text file.

    Parameters
    ----------
    file_path: str
        The path of the output file.
    entrez2gene: dict
        The mapping of Entrez IDs to gene symbols.

    Returns
    -------
    None

    """
    with misc.smart_open_write(file_path, mode='wb') as ofh:
        writer = csv.writer(ofh, dialect='excel-tab',
                            lineterminator=os.linesep)
        for k in sorted(entrez2gene.keys(), key=lambda x: int(x)):
            writer.writerow([k, entrez2gene[k]])
    logger.info('Output written to file "%s".', file_path)