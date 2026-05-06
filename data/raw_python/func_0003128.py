def lang_match_rdf(triple, accepted_languages):
    '''Find if the RDF triple contains acceptable language data'''
    if not accepted_languages:
        return True
    languages = set([n.language for n in triple if isinstance(n, Literal)])
    return (not languages) or (languages & accepted_languages)