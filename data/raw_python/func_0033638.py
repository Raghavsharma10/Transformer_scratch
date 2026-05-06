def process_key(string, key,
                fallback_sequence="", rules=None,
                skip_non_vietnamese=True):
    """Process a keystroke.

    Args:
        string: The previously processed string or "".
        key: The keystroke.
        fallback_sequence: The previous keystrokes.
        rules (optional): A dictionary listing
            transformation rules. Defaults to get_telex_definition().
        skip_non_vietnamese (optional): Whether to skip results that
            doesn't seem like Vietnamese. Defaults to True.

    Returns a tuple. The first item of which is the processed
    Vietnamese string, the second item is the next fallback sequence.
    The two items are to be fed back into the next call of process_key()
    as `string` and `fallback_sequence`. If `skip_non_vietnamese` is
    True and the resulting string doesn't look like Vietnamese,
    both items contain the `fallback_sequence`.

    >>> process_key('a', 'a', 'a')
    (â, aa)

    Note that when a key is an undo key, it won't get appended to
    `fallback_sequence`.

    >>> process_key('â', 'a', 'aa')
    (aa, aa)

    `rules` is a dictionary that maps keystrokes to
    their effect string. The effects can be one of the following:

    'a^': a with circumflex (â), only affect an existing 'a family'
    'a+': a with breve (ă), only affect an existing 'a family'
    'e^': e with circumflex (ê), only affect an existing 'e family'
    'o^': o with circumflex (ô), only affect an existing 'o family'
    'o*': o with horn (ơ), only affect an existing 'o family'
    'd-': d with bar (đ), only affect an existing 'd'
    '/': acute (sắc), affect an existing vowel
    '\': grave (huyền), affect an existing vowel
    '?': hook (hỏi), affect an existing vowel
    '~': tilde (ngã), affect an existing vowel
    '.': dot (nặng), affect an existing vowel
    '<ư': append ư
    '<ơ': append ơ

    A keystroke entry can have multiple effects, in which case the
    dictionary entry's value should be a list of the possible
    effect strings. Although you should try to avoid this if
    you are defining a custom input method rule.
    """
    # TODO Figure out a way to remove the `string` argument. Perhaps only the
    #      key sequence is needed?
    def default_return():
        return string + key, fallback_sequence + key

    if rules is None:
        rules = get_telex_definition()

    comps = utils.separate(string)

    # if not _is_processable(comps):
    #     return default_return()

    # Find all possible transformations this keypress can generate
    trans_list = _get_transformation_list(
        key, rules, fallback_sequence)

    # Then apply them one by one
    new_comps = list(comps)
    for trans in trans_list:
        new_comps = _transform(new_comps, trans)

    if new_comps == comps:
        tmp = list(new_comps)

        # If none of the transformations (if any) work
        # then this keystroke is probably an undo key.
        if _can_undo(new_comps, trans_list):
            # The prefix "_" means undo.
            for trans in map(lambda x: "_" + x, trans_list):
                new_comps = _transform(new_comps, trans)

            # Undoing the w key with the TELEX input method with the
            # w:<ư extension requires some care.
            #
            # The input (ư, w) should be undone as w
            # on the other hand, (ư, uw) should return uw.
            #
            # _transform() is not aware of the 2 ways to generate
            # ư in TELEX and always think ư was created by uw.
            # Therefore, after calling _transform() to undo ư,
            # we always get ['', 'u', ''].
            #
            # So we have to clean it up a bit.
            def is_telex_like():
                return '<ư' in rules["w"]

            def undone_vowel_ends_with_u():
                return new_comps[1] and new_comps[1][-1].lower() == "u"

            def not_first_key_press():
                return len(fallback_sequence) >= 1

            def user_typed_ww():
                return (fallback_sequence[-1:]+key).lower() == "ww"

            def user_didnt_type_uww():
                return not (len(fallback_sequence) >= 2 and
                            fallback_sequence[-2].lower() == "u")

            if is_telex_like() and \
                    not_first_key_press() and \
                    undone_vowel_ends_with_u() and \
                    user_typed_ww() and \
                    user_didnt_type_uww():
                # The vowel part of new_comps is supposed to end with
                # u now. That u should be removed.
                new_comps[1] = new_comps[1][:-1]

        if tmp == new_comps:
            fallback_sequence += key
        new_comps = utils.append_comps(new_comps, key)
    else:
        fallback_sequence += key

    if skip_non_vietnamese is True and key.isalpha() and \
            not is_valid_combination(new_comps, final_form=False):
        result = fallback_sequence, fallback_sequence
    else:
        result = utils.join(new_comps), fallback_sequence

    return result