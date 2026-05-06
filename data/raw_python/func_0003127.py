def lang_match_json(row, hdr, accepted_languages):
    '''Find if the JSON row contains acceptable language data'''
    if not accepted_languages:
        return True
    languages = set([row[c].get('xml:lang') for c in hdr
                     if c in row and row[c]['type'] == 'literal'])
    return (not languages) or (languages & accepted_languages)