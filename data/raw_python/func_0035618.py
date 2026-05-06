def write(
        stream_fragments, stream, normalize=True,
        book=None, sources=None, names=None, mappings=None):
    """
    Given an iterable of stream fragments, write it to the stream object
    by using its write method.  Returns a 3-tuple, where the first
    element is the mapping, second element is the list of sources and
    the third being the original names referenced by the given fragment.

    Arguments:

    stream_fragments
        an iterable that only contains StreamFragments
    stream
        an io.IOBase compatible stream object
    normalize
        the default True setting will result in the mappings that were
        returned be normalized to the minimum form.  This will reduce
        the size of the generated source map at the expense of slightly
        lower quality.

        Also, if any of the subsequent arguments are provided (for
        instance, for the multiple calls to this function), the usage of
        the normalize flag is currently NOT supported.

        If multiple sets of outputs are to be produced, the recommended
        method is to chain all the stream fragments together before
        passing in.

    Advanced usage arguments

    book
        A Book instance; if none is provided an instance will be created
        from the default_book constructor.  The Bookkeeper instance is
        used for tracking the positions of rows and columns of the input
        stream.
    sources
        a Names instance for tracking sources; if None is provided, an
        instance will be created for internal use.
    names
        a Names instance for tracking names; if None is provided, an
        instance will be created for internal use.
    mappings
        a previously produced mappings.

    A stream fragment tuple must contain the following

    - The string to write to the stream
    - Original starting line of the string; None if not present
    - Original starting column fo the line; None if not present
    - Original string that this fragment represents (i.e. for the case
      where this string fragment was an identifier but got mangled into
      an alternative form); use None if this was not the case.
    - The source of the fragment.  If the first fragment is unspecified,
      the INVALID_SOURCE url will be used (i.e. about:invalid).  After
      that, a None value will be treated as the implicit value, and if
      NotImplemented is encountered, the INVALID_SOURCE url will be used
      also.

    If a number of stream_fragments are to be provided, common instances
    of Book (constructed via default_book) and Names (for sources and
    names) should be provided if they are not chained together.
    """

    def push_line():
        mappings.append([])
        book.keeper._sink_column = 0

    if names is None:
        names = Names()

    if sources is None:
        sources = Names()

    if book is None:
        book = default_book()

    if not isinstance(mappings, list):
        # note that
        mappings = []
        # finalize initial states; the most recent list (mappings[-1])
        # is the current line
        push_line()

    for chunk, lineno, colno, original_name, source in stream_fragments:
        # note that lineno/colno are assumed to be both provided or none
        # provided.
        lines = chunk.splitlines(True)
        for line in lines:
            stream.write(line)

            # Two separate checks are done.  As per specification, if
            # either lineno or colno are unspecified, it is assumed that
            # the segment is unmapped - append a termination (1-tuple)
            #
            # Otherwise, note that if this segment is the beginning of a
            # line, and that an implied source colno/linecol were
            # provided (i.e. value of 0), and that the string is empty,
            # it can be safely skipped, since it is an implied and
            # unmapped indentation

            if lineno is None or colno is None:
                mappings[-1].append((book.keeper.sink_column,))
            else:
                name_id = names.update(original_name)
                # this is a bit of a trick: an unspecified value (None)
                # will simply be treated as the implied value, hence 0.
                # However, a NotImplemented will be recorded and be
                # convereted to the invalid url at the end.
                source_id = sources.update(source) or 0

                if lineno:
                    # a new lineno is provided, apply it to the book and
                    # use the result as the written value.
                    book.keeper.source_line = lineno
                    source_line = book.keeper.source_line
                else:
                    # no change in offset, do not calculate and assume
                    # the value to be written is unchanged.
                    source_line = 0

                # if the provided colno is to be inferred, calculate it
                # based on the previous line length plus the previous
                # real source column value, otherwise standard value
                # for tracking.

                # the reason for using the previous lengths is simply
                # due to how the bookkeeper class does the calculation
                # on-demand, and that the starting column for the
                # _current_ text fragment can only be calculated using
                # what was written previously, hence the original length
                # value being added if the current colno is to be
                # inferred.
                if colno:
                    book.keeper.source_column = colno
                else:
                    book.keeper.source_column = (
                        book.keeper._source_column + book.original_len)

                if original_name is not None:
                    mappings[-1].append((
                        book.keeper.sink_column, source_id,
                        source_line, book.keeper.source_column,
                        name_id
                    ))
                else:
                    mappings[-1].append((
                        book.keeper.sink_column, source_id,
                        source_line, book.keeper.source_column
                    ))

            # doing this last to update the position for the next line
            # or chunk for the relative values based on what was added
            if line[-1:] in '\r\n':
                # Note: this HAS to be an edge case and should never
                # happen, but this has the potential to muck things up.
                # Since the parent only provided the start, will need
                # to manually track the chunks internal to here.
                # This normally shouldn't happen with sane parsers
                # and lexers, but this assumes that no further symbols
                # aside from the new lines got inserted.
                colno = (
                    colno if colno in (0, None) else
                    colno + len(line.rstrip()))
                book.original_len = book.written_len = 0
                push_line()

                if line is not lines[-1]:
                    logger.warning(
                        'text in the generated document at line %d may be '
                        'mapped incorrectly due to trailing newline character '
                        'in provided text fragment.', len(mappings)
                    )
                    logger.info(
                        'text in stream fragments should not have trailing '
                        'characters after a new line, they should be split '
                        'off into a separate fragment.'
                    )
            else:
                book.written_len = len(line)
                book.original_len = (
                    len(original_name) if original_name else book.written_len)
                book.keeper.sink_column = (
                    book.keeper._sink_column + book.written_len)

    # normalize everything
    if normalize:
        # if this _ever_ supports the multiple usage using existence
        # instances of names and book and mappings, it needs to deal
        # with NOT normalizing the existing mappings and somehow reuse
        # the previously stored value, probably in the book.  It is
        # most certainly a bad idea to support that use case while also
        # supporting the default normalize flag due to the complex
        # tracking of all the existing values...
        mappings = normalize_mappings(mappings)

    list_sources = [
        INVALID_SOURCE if s == NotImplemented else s for s in sources
    ] or [INVALID_SOURCE]
    return mappings, list_sources, list(names)