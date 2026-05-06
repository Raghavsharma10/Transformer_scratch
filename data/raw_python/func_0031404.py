def make_name_ac_map(assy_name, primary_only=False):
    """make map from sequence name to accession for given assembly name

    >>> grch38p5_name_ac_map = make_name_ac_map('GRCh38.p5')
    >>> grch38p5_name_ac_map['1']
    'NC_000001.11'

    """
    return {
        s['name']: s['refseq_ac']
        for s in get_assembly(assy_name)['sequences']
        if (not primary_only or _is_primary(s))
    }