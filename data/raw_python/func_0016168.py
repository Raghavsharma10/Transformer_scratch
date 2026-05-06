def shortname(inputid):  # type: (Text) -> Text
    """Returns the last segment of the provided fragment or path."""
    parsed_id = urllib.parse.urlparse(inputid)
    if parsed_id.fragment:
        return parsed_id.fragment.split(u"/")[-1]
    return parsed_id.path.split(u"/")[-1]