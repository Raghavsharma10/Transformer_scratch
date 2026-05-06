def write_sample_sheet(output_file, accessions, names, celfile_urls, sel=None):
    """Generate a sample sheet in tab-separated text format.

    The columns contain the following sample attributes:
    1) accession
    2) name
    3) CEL file name
    4) CEL file URL

    Parameters
    ----------
    output_file: str
        The path of the output file.
    accessions: list or tuple of str
        The sample accessions.
    names: list or tuple of str
        The sample names.
    celfile_urls: list or tuple of str
        The sample CEL file URLs.
    sel: Iterable, optional
        A list of sample indices to include. If None, all samples are included.
        [None]

    Returns
    -------
    None
    """
    assert isinstance(output_file, str)
    assert isinstance(accessions, (list, tuple))
    for acc in accessions:
        assert isinstance(acc, str)
    assert isinstance(names, (list, tuple))
    for n in names:
        assert isinstance(n, str)
    assert isinstance(celfile_urls, (list, tuple))
    for u in celfile_urls:
        assert isinstance(u, str)
    if sel is not None:
        assert isinstance(sel, Iterable)
        for i in sel:
            assert isinstance(i, (int, np.integer))

    with open(output_file, 'wb') as ofh:
        writer = csv.writer(ofh, dialect='excel-tab',
                            lineterminator=os.linesep,
                            quoting=csv.QUOTE_NONE)
        # write header
        writer.writerow(['Accession', 'Name', 'CEL file name', 'CEL file URL'])
        n = len(list(names))
        if sel is None:
            sel = range(n)
        for i in sel:
            cf = celfile_urls[i].split('/')[-1]
            # row = [accessions[i], names[i], cf, celfile_urls[i]]
            writer.writerow([accessions[i], names[i], cf, celfile_urls[i]])