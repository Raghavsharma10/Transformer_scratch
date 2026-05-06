def get_assembly(name):
    """read a single assembly by name, returning a dictionary of assembly data

    >>> assy = get_assembly('GRCh37.p13')

    >>> assy['name']
    'GRCh37.p13'

    >>> assy['description']
    'Genome Reference Consortium Human Build 37 patch release 13 (GRCh37.p13)'

    >>> assy['refseq_ac']
    'GCF_000001405.25'

    >>> assy['genbank_ac']
    'GCA_000001405.14'

    >>> len(assy['sequences'])
    297

    >>> import pprint
    >>> pprint.pprint(assy['sequences'][0])
    {'aliases': ['chr1'],
     'assembly_unit': 'Primary Assembly',
     'genbank_ac': 'CM000663.1',
     'length': 249250621,
     'name': '1',
     'refseq_ac': 'NC_000001.10',
     'relationship': '=',
     'sequence_role': 'assembled-molecule'}
    """

    fn = pkg_resources.resource_filename(
        __name__, _assy_path_fmt.format(name=name))
    return json.load(gzip.open(fn, mode="rt", encoding="utf-8"))