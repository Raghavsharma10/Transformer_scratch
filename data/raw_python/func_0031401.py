def get_assembly_names():
    """return list of available assemblies
    
    >>> assy_names = get_assembly_names()

    >>> 'GRCh37.p13' in assy_names
    True

    """

    return [
        n.replace(".json.gz", "")
        for n in pkg_resources.resource_listdir(__name__, _assy_dir)
        if n.endswith(".json.gz")
    ]