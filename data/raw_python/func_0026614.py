def parser(parser_type=basic_parser, functions=None, patterns=None, expressions=None, patterns_yaml_path=None,
           expressions_yaml_path=None):
    """ A Reparse parser description.
        Simply provide the functions, patterns, & expressions to build.
        If you are using YAML for expressions + patterns, you can use
        ``expressions_yaml_path`` & ``patterns_yaml_path`` for convenience.

        The default parser_type is the basic ordered parser.
    """
    from reparse.builders import build_all
    from reparse.validators import validate

    def _load_yaml(file_path):
        import yaml
        with open(file_path) as f:
            return yaml.safe_load(f)

    assert expressions or expressions_yaml_path, "Reparse can't build a parser without expressions"
    assert patterns or patterns_yaml_path, "Reparse can't build a parser without patterns"
    assert functions, "Reparse can't build without a functions"

    if patterns_yaml_path:
        patterns = _load_yaml(patterns_yaml_path)
    if expressions_yaml_path:
        expressions = _load_yaml(expressions_yaml_path)
    validate(patterns, expressions)

    return parser_type(build_all(patterns, expressions, functions))