def insert_rawblocks_factory(rawblocks):
    r"""Returns insert_rawblocks(key, value, fmt, meta) action that inserts
    non-duplicate RawBlock elements.
    """

    # pylint: disable=unused-argument
    def insert_rawblocks(key, value, fmt, meta):
        """Inserts non-duplicate RawBlock elements."""

        if not rawblocks:
            return None

        # Put the RawBlock elements in front of the first block element that
        # isn't also a RawBlock.

        if not key in ['Plain', 'Para', 'CodeBlock', 'RawBlock',
                       'BlockQuote', 'OrderedList', 'BulletList',
                       'DefinitionList', 'Header', 'HorizontalRule',
                       'Table', 'Div', 'Null']:
            return None

        if key == 'RawBlock':  # Remove duplicates
            rawblock = RawBlock(*value)
            if rawblock in rawblocks:
                rawblocks.remove(rawblock)
                return None

        if rawblocks:  # Insert blocks
            el = _getel(key, value)
            return [rawblocks.pop(0) for i in range(len(rawblocks))] + [el]

        return None

    return insert_rawblocks