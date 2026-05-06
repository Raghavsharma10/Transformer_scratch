def dedent(s):
    """Removes the hanging dedent from all the first line of a string."""
    head, _, tail = s.partition('\n')
    dedented_tail = textwrap.dedent(tail)
    result = "{head}\n{tail}".format(
        head=head,
        tail=dedented_tail)
    return result