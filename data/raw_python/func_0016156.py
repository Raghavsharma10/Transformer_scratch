def add_namespaces(metadata, namespaces):
    # type: (Mapping[Text, Any], MutableMapping[Text, Text]) -> None
    """Collect the provided namespaces, checking for conflicts."""
    for key, value in metadata.items():
        if key not in namespaces:
            namespaces[key] = value
        elif namespaces[key] != value:
            raise validate.ValidationException(
                "Namespace prefix '{}' has conflicting definitions '{}'"
                " and '{}'.".format(key, namespaces[key], value))