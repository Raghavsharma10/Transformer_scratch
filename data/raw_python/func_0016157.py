def collect_namespaces(metadata):
    # type: (Mapping[Text, Any]) -> Dict[Text, Text]
    """Walk through the metadata object, collecting namespace declarations."""
    namespaces = {}  # type: Dict[Text, Text]
    if "$import_metadata" in metadata:
        for value in metadata["$import_metadata"].values():
            add_namespaces(collect_namespaces(value), namespaces)
    if "$namespaces" in metadata:
        add_namespaces(metadata["$namespaces"], namespaces)
    return namespaces