def get_cytoband_maps(names=[]):
    """Load all cytoband maps

    >>> maps = get_cytoband_maps()
    >>> maps["ucsc-hg38"]["1"]["p32.2"]
    [55600000, 58500000, 'gpos50']
    >>> maps["ucsc-hg19"]["1"]["p32.2"]
    [56100000, 59000000, 'gpos50']
    """
    if names == []:
        names = get_cytoband_names()
    return {name: get_cytoband_map(name) for name in names}