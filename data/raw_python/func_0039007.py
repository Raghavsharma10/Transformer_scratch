def process_refs_factory(labels):
    """Returns process_refs(key, value, fmt, meta) action that processes
    text around a reference.  Only references with labels found in the
    'labels' list are processed.

    Consider the markdown "{+@fig:1}", which represents a reference to a
    figure. "@" denotes a reference, "fig:1" is the reference's label, and
    "+" is a modifier.  Valid modifiers are '+, '*' and '!'.

    This function strips curly braces and adds the modifiers to the attributes
    of Cite elements.  Cite attributes must be detached before the document is
    written to STDOUT because pandoc doesn't recognize them.  Alternatively,
    use an action from replace_refs_factory() to replace the references
    altogether.
    """

    # pylint: disable=unused-argument
    def process_refs(key, value, fmt, meta):
        """Instates Ref elements."""
        # References may occur in a variety of places; we must process them
        # all.

        if key in ['Para', 'Plain']:
            _process_refs(value, labels)
        elif key == 'Image':
            _process_refs(value[-2], labels)
        elif key == 'Table':
            _process_refs(value[-5], labels)
        elif key == 'Span':
            _process_refs(value[-1], labels)
        elif key == 'Emph':
            _process_refs(value, labels)
        elif key == 'Strong':
            _process_refs(value, labels)

    return process_refs