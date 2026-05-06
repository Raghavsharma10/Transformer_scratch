def _parse_relation(chunk, type="O"):
    """ Returns a string of the roles and relations parsed from the given <chunk> element.
        The chunk type (which is part of the relation string) can be given as parameter.
    """
    r1 = chunk.get(XML_RELATION)
    r2 = chunk.get(XML_ID, chunk.get(XML_OF))
    r1 = [x != "-" and x or None for x in r1.split("|")] or [None]
    r2 = [x != "-" and x or None for x in r2.split("|")] or [None]
    r2 = [x is not None and x.split(_UID_SEPARATOR )[-1] or x for x in r2]
    if len(r1) < len(r2): r1 = r1 + r1 * (len(r2)-len(r1)) # [1] ["SBJ", "OBJ"] => "SBJ-1;OBJ-1"
    if len(r2) < len(r1): r2 = r2 + r2 * (len(r1)-len(r2)) # [2,4] ["OBJ"] => "OBJ-2;OBJ-4"
    return ";".join(["-".join([x for x in (type, r1, r2) if x]) for r1, r2 in zip(r1, r2)])