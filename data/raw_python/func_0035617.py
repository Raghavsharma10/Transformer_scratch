def normalize_mapping_line(mapping_line, previous_source_column=0):
    """
    Often times the position will remain stable, such that the naive
    process will end up with many redundant values; this function will
    iterate through the line and remove all extra values.
    """

    if not mapping_line:
        return [], previous_source_column

    # Note that while the local record here is also done as a 4-tuple,
    # element 1 and 2 are never used since they are always provided by
    # the segments in the mapping line; they are defined for consistency
    # reasons.

    def regenerate(segment):
        if len(segment) == 5:
            result = (record[0], segment[1], segment[2], record[3], segment[4])
        else:
            result = (record[0], segment[1], segment[2], record[3])
        # Ideally the exact location should still be kept, but given
        # that the sourcemap format is accumulative and permits a lot
        # of inferred positions, resetting all values to 0 is intended.
        record[:] = [0, 0, 0, 0]
        return result

    # first element of the line; sink column (0th element) is always
    # the absolute value, so always use the provided value sourced from
    # the original mapping_line; the source column (3rd element) is
    # never reset, so if a previous counter exists (which is specified
    # by the optional argument), make use of it to generate the initial
    # normalized segment.
    record = [0, 0, 0, previous_source_column]
    result = []
    regen_next = True

    for segment in mapping_line:
        if not segment:
            # ignore empty records
            continue
        # if the line has not changed, and that the increases of both
        # columns are the same, accumulate the column counter and drop
        # the segment.

        # accumulate the current record first
        record[0] += segment[0]
        if len(segment) == 1:
            # Mark the termination, as 1-tuple determines the end of the
            # previous symbol and denote that whatever follows are not
            # in any previous source files.  So if it isn't recorded,
            # make note of this if it wasn't done already.
            if result and len(result[-1]) != 1:
                result.append((record[0],))
                record[0] = 0
                # the next complete segment will require regeneration
                regen_next = True
            # skip the remaining processing.
            continue

        record[3] += segment[3]

        # 5-tuples are always special case with the remapped identifier
        # name element, and to mark the termination the next token must
        # also be explicitly written (in our case, regenerated).  If the
        # filename or source line relative position changed (idx 1 and
        # 2), regenerate it too.  Finally, if the column offsets differ
        # between source and sink, regenerate.
        if len(segment) == 5 or regen_next or segment[1] or segment[2] or (
                record[0] != record[3]):
            result.append(regenerate(segment))
            regen_next = len(segment) == 5

    # must return the consumed/omitted values.
    return result, record[3]