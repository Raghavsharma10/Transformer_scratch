def get_cytoband_map(name):
    """Fetch one cytoband map by name

    >>> map = get_cytoband_map("ucsc-hg38")
    >>> map["1"]["p32.2"]
    [55600000, 58500000, 'gpos50']

    """
    fn = pkg_resources.resource_filename(
        __name__, _data_path_fmt.format(name=name))
    return json.load(gzip.open(fn, mode="rt", encoding="utf-8"))