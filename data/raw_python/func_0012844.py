def makename2refdct(commdct):
    """make the name2refs dict in the idd_index"""
    refdct = {}
    for comm in commdct: # commdct is a list of dict
        try:
            idfobj = comm[0]['idfobj'].upper()
            field1 = comm[1]
            if 'Name' in field1['field']:
                references = field1['reference']
                refdct[idfobj] = references
        except (KeyError, IndexError) as e:
            continue # not the expected pattern for reference
    return refdct