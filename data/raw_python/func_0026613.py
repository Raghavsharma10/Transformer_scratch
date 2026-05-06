def build_tree_parser(patterns):
    """ This parser_type simply outputs an array of [(tree, regex)]
        for use in another language.
    """
    def output():
        for pattern in patterns:
            yield (pattern.build_full_tree(), pattern.regex)
    return list(output())