def alt_parser(patterns):
    """ This parser is able to handle multiple different patterns
        finding stuff in text-- while removing matches that overlap.
    """
    from reparse.util import remove_lower_overlapping
    get_first = lambda items: [i[0] for i in items]
    get_second = lambda items: [i[1] for i in items]

    def parse(line):
        output = []
        for pattern in patterns:
            results = pattern.scan(line)
            if results and any(results):
                output.append((pattern.order, results))
        return get_first(reduce(remove_lower_overlapping, get_second(sorted(output)), []))

    return parse