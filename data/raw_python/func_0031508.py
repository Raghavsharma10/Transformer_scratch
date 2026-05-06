def get_argument_parser(prog=None, desc=None, formatter_class=None):
    """Create an argument parser.

    Parameters
    ----------
    prog: str
        The program name.
    desc: str
        The program description.
    formatter_class: argparse formatter class, optional
        The argparse formatter class to use.

    Returns
    -------
    `argparse.ArgumentParser`
        The arguemnt parser created.
    """
    if formatter_class is None:
        formatter_class = argparse.RawTextHelpFormatter

    parser = argparse.ArgumentParser(
        prog=prog, description=desc,
        formatter_class=formatter_class, add_help=False
    )

    g = parser.add_argument_group('Help')
    g.add_argument('-h', '--help', action='help',
                   help='Show this help message and exit.')

    v = genometools.__version__
    g.add_argument('--version', action='version', version='GenomeTools ' + v,
                   help='Output the GenomeTools version and exit.')

    return parser