def and_terms(*args):
    """ Connect given term strings or list(s) of term strings with an AND operator for querying.

        Args:
            An arbitrary number of either strings or lists of strings representing query terms.

        Returns
            A query string consisting of argument terms and'ed together.
    """
    args = [arg if not isinstance(arg, list) else ' '.join(arg) for arg in args]
    return '({0})'.format(' '.join(args))