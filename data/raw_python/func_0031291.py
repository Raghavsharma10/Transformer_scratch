def get_cytoband_names():
    """Returns the names of available cytoband data files

    >> get_cytoband_names()
    ['ucsc-hg38', 'ucsc-hg19']
    """
    return [
        n.replace(".json.gz", "")
        for n in pkg_resources.resource_listdir(__name__, _data_dir)
        if n.endswith(".json.gz")
    ]