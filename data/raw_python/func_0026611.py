def basic_parser(patterns, with_name=None):
    """ Basic ordered parser.
    """
    def parse(line):
        output = None
        highest_order = 0
        highest_pattern_name = None
        for pattern in patterns:
            results = pattern.findall(line)
            if results and any(results):
                if pattern.order > highest_order:
                    output = results
                    highest_order = pattern.order
                    if with_name:
                        highest_pattern_name = pattern.name
        if with_name:
            return output, highest_pattern_name
        return output

    return parse