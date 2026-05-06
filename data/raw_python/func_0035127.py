def ParserUnparserFactory(module_name, *unparser_names):
    """
    Produce a new parser/unparser object from the names provided.
    """

    parse_callable = import_module(PKGNAME + '.parsers.' + module_name).parse
    unparser_module = import_module(PKGNAME + '.unparsers.' + module_name)
    return RawParserUnparserFactory(module_name, parse_callable, *[
        getattr(unparser_module, name) for name in unparser_names])