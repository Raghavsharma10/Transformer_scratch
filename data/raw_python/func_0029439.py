def convert_parser_to(parser, parser_or_type, metadata_props=None):
    """
    :return: a parser of type parser_or_type, initialized with the properties of parser. If parser_or_type
    is a type, an instance of it must contain a update method. The update method must also process
    the set of properties supported by MetadataParser for the conversion to have any affect.
    :param parser: the parser (or content or parser type) to convert to new_type
    :param parser_or_type: a parser (or content) or type of parser to return
    :see: get_metadata_parser(metadata_container) for more on how parser_or_type is treated
    """

    old_parser = parser if isinstance(parser, MetadataParser) else get_metadata_parser(parser)
    new_parser = get_metadata_parser(parser_or_type)

    for prop in (metadata_props or _supported_props):
        setattr(new_parser, prop, deepcopy(getattr(old_parser, prop, u'')))

    new_parser.update()

    return new_parser