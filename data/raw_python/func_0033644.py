def handle_backspace(converted_string, raw_sequence, im_rules=None):
    """
    Returns a new raw_sequence after a backspace. This raw_sequence should
    be pushed back to process_sequence().
    """
    # I can't find a simple explanation for this, so
    # I hope this example can help clarify it:
    #
    # handle_backspace(thương, thuwongw) -> thuwonw
    # handle_backspace(thươn, thuwonw) -> thuwow
    # handle_backspace(thươ, thuwow) -> thuw
    # handle_backspace(thươ, thuw) -> th
    #
    # The algorithm for handle_backspace was contributed by @hainp.

    if im_rules == None:
        im_rules = get_telex_definition()

    deleted_char = converted_string[-1]

    _accent = accent.get_accent_char(deleted_char)
    _mark = mark.get_mark_char(deleted_char)

    if _mark or _accent:
        # Find a sequence of IM keys at the end of
        # raw_sequence

        ime_keys_at_end = ""
        len_raw_sequence = len(raw_sequence)
        i = len_raw_sequence - 1

        while i >= 0:
            if raw_sequence[i] not in im_rules and \
                    raw_sequence[i] not in "aeiouyd":
                i += 1
                break
            else:
                ime_keys_at_end = raw_sequence[i] + ime_keys_at_end
            i -= 1

        # Try to find a subsequence from that sequence
        # that can be converted to the deleted_char
        k = 0
        while k < len_raw_sequence:
            if process_sequence(raw_sequence[i + k:], im_rules) == deleted_char:
                # Delete that subsequence
                raw_sequence = raw_sequence[:i + k]
                break
            k += 1
    else:
        index = raw_sequence.rfind(deleted_char)
        raw_sequence = raw_sequence[:index] + raw_sequence[(index + 1):]

    return raw_sequence