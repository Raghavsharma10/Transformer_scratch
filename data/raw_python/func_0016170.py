def print_fieldrefs(doc, loader, stream):
    # type: (List[Dict[Text, Any]], Loader, IO) -> None
    """Write a GraphViz graph of the relationships between the fields."""
    obj = extend_and_specialize(doc, loader)

    primitives = set(("http://www.w3.org/2001/XMLSchema#string",
                      "http://www.w3.org/2001/XMLSchema#boolean",
                      "http://www.w3.org/2001/XMLSchema#int",
                      "http://www.w3.org/2001/XMLSchema#long",
                      "https://w3id.org/cwl/salad#null",
                      "https://w3id.org/cwl/salad#enum",
                      "https://w3id.org/cwl/salad#array",
                      "https://w3id.org/cwl/salad#record",
                      "https://w3id.org/cwl/salad#Any"))

    stream.write("digraph {\n")
    for entry in obj:
        if entry.get("abstract"):
            continue
        if entry["type"] == "record":
            label = shortname(entry["name"])
            for field in entry.get("fields", []):
                found = set()  # type: Set[Text]
                field_name = shortname(field["name"])
                replace_type(field["type"], {}, loader, found, find_embeds=False)
                for each_type in found:
                    if each_type not in primitives:
                        stream.write(
                            "\"%s\" -> \"%s\" [label=\"%s\"];\n"
                            % (label, shortname(each_type), field_name))
    stream.write("}\n")