def get_parser():
    """Get a parser object"""
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description=__doc__,
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-s1", dest="s1", help="sequence 1")
    parser.add_argument("-s2", dest="s2", help="sequence 2")
    return parser