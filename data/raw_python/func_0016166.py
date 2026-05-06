def extend_and_specialize(items, loader):
    # type: (List[Dict[Text, Any]], Loader) -> List[Dict[Text, Any]]
    """
    Apply 'extend' and 'specialize' to fully materialize derived record types.
    """

    items = deepcopy_strip(items)
    types = {i["name"]: i for i in items}  # type: Dict[Text, Any]
    results = []

    for stype in items:
        if "extends" in stype:
            specs = {}  # type: Dict[Text, Text]
            if "specialize" in stype:
                for spec in aslist(stype["specialize"]):
                    specs[spec["specializeFrom"]] = spec["specializeTo"]

            exfields = []  # type: List[Text]
            exsym = []  # type: List[Text]
            for ex in aslist(stype["extends"]):
                if ex not in types:
                    raise Exception(
                        "Extends {} in {} refers to invalid base type.".format(
                            stype["extends"], stype["name"]))

                basetype = copy.copy(types[ex])

                if stype["type"] == "record":
                    if specs:
                        basetype["fields"] = replace_type(
                            basetype.get("fields", []), specs, loader, set())

                    for field in basetype.get("fields", []):
                        if "inherited_from" not in field:
                            field["inherited_from"] = ex

                    exfields.extend(basetype.get("fields", []))
                elif stype["type"] == "enum":
                    exsym.extend(basetype.get("symbols", []))

            if stype["type"] == "record":
                stype = copy.copy(stype)
                exfields.extend(stype.get("fields", []))
                stype["fields"] = exfields

                fieldnames = set()  # type: Set[Text]
                for field in stype["fields"]:
                    if field["name"] in fieldnames:
                        raise validate.ValidationException(
                            "Field name {} appears twice in {}".format(
                                field["name"], stype["name"]))
                    else:
                        fieldnames.add(field["name"])
            elif stype["type"] == "enum":
                stype = copy.copy(stype)
                exsym.extend(stype.get("symbols", []))
                stype["symbol"] = exsym

            types[stype["name"]] = stype

        results.append(stype)

    ex_types = {}
    for result in results:
        ex_types[result["name"]] = result

    extended_by = {}  # type: Dict[Text, Text]
    for result in results:
        if "extends" in result:
            for ex in aslist(result["extends"]):
                if ex_types[ex].get("abstract"):
                    add_dictlist(extended_by, ex, ex_types[result["name"]])
                    add_dictlist(extended_by, avro_name(ex), ex_types[ex])

    for result in results:
        if result.get("abstract") and result["name"] not in extended_by:
            raise validate.ValidationException(
                "{} is abstract but missing a concrete subtype".format(
                    result["name"]))

    for result in results:
        if "fields" in result:
            result["fields"] = replace_type(
                result["fields"], extended_by, loader, set())

    return results