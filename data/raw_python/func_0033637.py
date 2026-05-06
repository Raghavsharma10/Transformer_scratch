def process_sequence(sequence,
                     rules=None,
                     skip_non_vietnamese=True):
    """\
    Convert a key sequence into a Vietnamese string with diacritical marks.

    Args:
        rules (optional): see docstring for process_key().
        skip_non_vietnamese (optional): see docstring for process_key().

    It even supports continous key sequences connected by separators.
    i.e. process_sequence('con meof.ddieen') should work.
    """
    result = ""
    raw = result
    result_parts = []
    if rules is None:
        rules = get_telex_definition()

    accepted_chars = _accepted_chars(rules)

    for key in sequence:
        if key not in accepted_chars:
            result_parts.append(result)
            result_parts.append(key)
            result = ""
            raw = ""
        else:
            result, raw = process_key(
                string=result,
                key=key,
                fallback_sequence=raw,
                rules=rules,
                skip_non_vietnamese=skip_non_vietnamese)

    result_parts.append(result)
    return ''.join(result_parts)