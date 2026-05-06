def fw_romaji_lt(full, regular):
    '''
    Generates a lookup table with the fullwidth rōmaji characters
    on the left side, and the regular rōmaji characters as the values.
    '''
    lt = {}
    for n in range(len(full)):
        fw = full[n]
        reg = regular[n]
        lt[fw] = reg

    return lt