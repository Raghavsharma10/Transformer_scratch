def yaml_dump(data, stream=None):
    # type: (YamlData, Optional[TextIO]) -> Text
    """ Dump data to a YAML string/file.

    Args:
        data (YamlData):
            The data to serialize as YAML.
        stream (TextIO):
            The file-like object to save to. If given, this function will write
            the resulting YAML to that stream.

    Returns:
        str: The YAML string.
    """
    return yaml.dump(
        data,
        stream=stream,
        Dumper=Dumper,
        default_flow_style=False
    )