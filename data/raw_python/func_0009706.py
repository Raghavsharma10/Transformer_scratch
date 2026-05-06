def defineID(defid):
    """Search for UD's definition ID and return list of UrbanDefinition objects.

    Keyword arguments:
    defid -- definition ID to search for (int or str)
    """
    json = _get_urban_json(UD_DEFID_URL + urlquote(str(defid)))
    return _parse_urban_json(json)