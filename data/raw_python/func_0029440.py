def get_metadata_parser(metadata_container, **metadata_defaults):
    """
    Takes a metadata_container, which may be a type or instance of a parser, a dict, string, or file.
    :return: a new instance of a parser corresponding to the standard represented by metadata_container
    :see: get_parsed_content(metdata_content) for more on types of content that can be parsed
    """

    parser_type = None

    if isinstance(metadata_container, MetadataParser):
        parser_type = type(metadata_container)

    elif isinstance(metadata_container, type):
        parser_type = metadata_container
        metadata_container = metadata_container().update(**metadata_defaults)

    xml_root, xml_tree = get_parsed_content(metadata_container)

    # The get_parsed_content method ensures only these roots will be returned

    parser = None

    if parser_type is not None:
        parser = parser_type(xml_tree, **metadata_defaults)
    elif xml_root in ISO_ROOTS:
        parser = IsoParser(xml_tree, **metadata_defaults)
    else:
        has_arcgis_data = any(element_exists(xml_tree, e) for e in ARCGIS_NODES)

        if xml_root == FGDC_ROOT and not has_arcgis_data:
            parser = FgdcParser(xml_tree, **metadata_defaults)
        elif xml_root in ARCGIS_ROOTS:
            parser = ArcGISParser(xml_tree, **metadata_defaults)

    return parser