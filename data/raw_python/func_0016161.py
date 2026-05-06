def get_anon_name(rec):
    # type: (MutableMapping[Text, Any]) -> Text
    """Calculate a reproducible name for anonymous types."""
    if "name" in rec:
        return rec["name"]
    anon_name = ""
    if rec['type'] in ('enum', 'https://w3id.org/cwl/salad#enum'):
        for sym in rec["symbols"]:
            anon_name += sym
        return "enum_"+hashlib.sha1(anon_name.encode("UTF-8")).hexdigest()
    if rec['type'] in ('record', 'https://w3id.org/cwl/salad#record'):
        for field in rec["fields"]:
            anon_name += field["name"]
        return "record_"+hashlib.sha1(anon_name.encode("UTF-8")).hexdigest()
    if rec['type'] in ('array', 'https://w3id.org/cwl/salad#array'):
        return ""
    raise validate.ValidationException("Expected enum or record, was %s" % rec['type'])