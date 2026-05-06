def sonseq(word):
    '''Return True if 'word' does not violate sonority sequencing.'''
    parts = re.split(r'([ieaouäöy]+)', word, flags=re.I | re.U)
    onset, coda = parts[0], parts[-1]

    #  simplex onset      Finnish complex onset
    if len(onset) <= 1 or onset.lower() in ONSETS:
        #      simplex coda    Finnish complex coda
        return len(coda) <= 1  # or coda in codas_inventory

    return False