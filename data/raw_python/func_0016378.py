def hunkify_lines(lines, context=3):
    """
    Return a list of line hunks given a list of lines `lines`. The number of
    context lines can be control with `context` which will return line hunks
    surrounded with `context` lines before and after the code change.
    """
    # Find contiguous line changes
    ranges = []
    range_start = None
    for i, line in enumerate(lines):
        if line.status is not None:
            if range_start is None:
                range_start = i
                continue
        elif range_start is not None:
            range_stop = i
            ranges.append((range_start, range_stop))
            range_start = None
    else:
        # Append the last range
        if range_start is not None:
            range_stop = i
            ranges.append((range_start, range_stop))

    # add context
    ranges_w_context = []
    for range_start, range_stop in ranges:
        range_start = range_start - context
        range_start = range_start if range_start >= 0 else 0
        range_stop = range_stop + context
        ranges_w_context.append((range_start, range_stop))

    # merge overlapping hunks
    merged_ranges = ranges_w_context[:1]
    for range_start, range_stop in ranges_w_context[1:]:
        prev_start, prev_stop = merged_ranges[-1]
        if range_start <= prev_stop:
            range_start = prev_start
            merged_ranges[-1] = (range_start, range_stop)
        else:
            merged_ranges.append((range_start, range_stop))

    # build final hunks
    hunks = []
    for range_start, range_stop in merged_ranges:
        hunk = lines[range_start:range_stop]
        hunks.append(hunk)

    return hunks