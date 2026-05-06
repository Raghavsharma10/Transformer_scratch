def make_valid_avro(items,          # type: Avro
                    alltypes,       # type: Dict[Text, Dict[Text, Any]]
                    found,          # type: Set[Text]
                    union=False     # type: bool
                    ):
    # type: (...) -> Union[Avro, Dict, Text]
    """Convert our schema to be more avro like."""
    # Possibly could be integrated into our fork of avro/schema.py?
    if isinstance(items, MutableMapping):
        items = copy.copy(items)
        if items.get("name") and items.get("inVocab", True):
            items["name"] = avro_name(items["name"])

        if "type" in items and items["type"] in (
                "https://w3id.org/cwl/salad#record",
                "https://w3id.org/cwl/salad#enum", "record", "enum"):
            if (hasattr(items, "get") and items.get("abstract")) or ("abstract"
                                                                     in items):
                return items
            if items["name"] in found:
                return cast(Text, items["name"])
            found.add(items["name"])
        for field in ("type", "items", "values", "fields"):
            if field in items:
                items[field] = make_valid_avro(
                    items[field], alltypes, found, union=True)
        if "symbols" in items:
            items["symbols"] = [avro_name(sym) for sym in items["symbols"]]
        return items
    if isinstance(items, MutableSequence):
        ret = []
        for i in items:
            ret.append(make_valid_avro(i, alltypes, found, union=union))  # type: ignore
        return ret
    if union and isinstance(items, string_types):
        if items in alltypes and avro_name(items) not in found:
            return cast(Dict, make_valid_avro(alltypes[items], alltypes, found,
                                              union=union))
        items = avro_name(items)
    return items