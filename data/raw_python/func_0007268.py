def read_lines(in_file):
    """Returns a list of lines from a input markdown file."""

    with open(in_file, 'r') as inf:
        in_contents = inf.read().split('\n')
    return in_contents