def addModlist(entry: dict, ignore_attr_types: Optional[List[str]] = None) -> Dict[str, List[bytes]]:
    """Build modify list for call of method LDAPObject.add()"""
    ignore_attr_types = _list_dict(map(str.lower, (ignore_attr_types or [])))
    modlist: Dict[str, List[bytes]] = {}
    for attrtype in entry.keys():
        if attrtype.lower() in ignore_attr_types:
            # This attribute type is ignored
            continue
        for value in entry[attrtype]:
            assert value is not None
        if len(entry[attrtype]) > 0:
            modlist[attrtype] = escape_list(entry[attrtype])
    return modlist