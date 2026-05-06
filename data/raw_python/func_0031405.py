def make_ac_name_map(assy_name, primary_only=False):
    """make map from accession (str) to sequence name (str) for given assembly name

    >>> grch38p5_ac_name_map = make_ac_name_map('GRCh38.p5')
    >>> grch38p5_ac_name_map['NC_000001.11']
    '1'

    """

    return {
        s['refseq_ac']: s['name']
        for s in get_assembly(assy_name)['sequences']
        if (not primary_only or _is_primary(s))
    }