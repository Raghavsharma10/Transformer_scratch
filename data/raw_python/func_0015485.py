def normalize_startparm(p: STARTPARM) -> List[Union[type(START), START_TYPE, URIRef]]:
    """ Return the startspec for p """
    if not isinstance(p, list):
        p = [p]
    return [normalize_uri(e) if isinstance(e, str) and e is not START else e for e in p]