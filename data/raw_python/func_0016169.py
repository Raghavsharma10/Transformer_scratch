def print_inheritance(doc, stream):
    # type: (List[Dict[Text, Any]], IO) -> None
    """Write a Grapviz inheritance graph for the supplied document."""
    stream.write("digraph {\n")
    for entry in doc:
        if entry["type"] == "record":
            label = name = shortname(entry["name"])
            fields = entry.get("fields", [])
            if fields:
                label += "\\n* %s\\l" % (
                    "\\l* ".join(shortname(field["name"])
                                 for field in fields))
            shape = "ellipse" if entry.get("abstract") else "box"
            stream.write("\"%s\" [shape=%s label=\"%s\"];\n"
                         % (name, shape, label))
            if "extends" in entry:
                for target in aslist(entry["extends"]):
                    stream.write("\"%s\" -> \"%s\";\n"
                                 % (shortname(target), name))
    stream.write("}\n")