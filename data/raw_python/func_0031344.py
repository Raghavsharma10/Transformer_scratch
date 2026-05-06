def read_series_matrix(path, encoding):
    """Read the series matrix."""
    assert isinstance(path, str)

    accessions = None
    titles = None
    celfile_urls = None
    with misc.smart_open_read(path, mode='rb', try_gzip=True) as fh:
        reader = csv.reader(fh, dialect='excel-tab', encoding=encoding)
        for l in reader:
            if not l:
                continue
            if l[0] == '!Sample_geo_accession':
                accessions = l[1:]
            elif l[0] == '!Sample_title':
                titles = l[1:]
            elif l[0] == '!Sample_supplementary_file' and celfile_urls is None:
                celfile_urls = l[1:]
            elif l[0] == '!series_matrix_table_begin':
                # we've read the end of the section containing metadata
                break
    return accessions, titles, celfile_urls