def width_aware_slice(s, start, end, replacement_char=u' '):
    # type: (Text, int, int, Text)
    """
    >>> width_aware_slice(u'a\uff25iou', 0, 2)[1] == u' '
    True
    """
    divides = [0]
    for c in s:
        divides.append(divides[-1] + wcswidth(c))

    new_chunk_chars = []
    for char, char_start, char_end in zip(s, divides[:-1], divides[1:]):
        if char_start == start and char_end == start:
            continue  # don't use zero-width characters at the beginning of a slice
                      # (combining characters combine with the chars before themselves)
        elif char_start >= start and char_end <= end:
            new_chunk_chars.append(char)
        else:
            new_chunk_chars.extend(replacement_char * interval_overlap(char_start, char_end, start, end))

    return ''.join(new_chunk_chars)