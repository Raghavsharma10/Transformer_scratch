def wrap(text, width, *args, **kwargs):
    """
    Like :func:`textwrap.wrap` but preserves existing newlines which
    :func:`textwrap.wrap` does not otherwise handle well.

    See Also
    --------
    :func:`textwrap.wrap`
    """

    return sum([textwrap.wrap(line, width, *args, **kwargs)
                if line else [''] for line in text.splitlines()], [])