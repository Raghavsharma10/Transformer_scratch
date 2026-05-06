def _transform(comps, trans):
    """
    Transform the given string with transform type trans
    """
    logging.debug("== In _transform(%s, %s) ==", comps, trans)
    components = list(comps)

    action, parameter = _get_action(trans)
    if action == _Action.ADD_MARK and \
            components[2] == "" and \
            mark.strip(components[1]).lower() in ['oe', 'oa'] and trans == "o^":
        action, parameter = _Action.ADD_CHAR, trans[0]

    if action == _Action.ADD_ACCENT:
        logging.debug("add_accent(%s, %s)", components, parameter)
        components = accent.add_accent(components, parameter)
    elif action == _Action.ADD_MARK and mark.is_valid_mark(components, trans):
        logging.debug("add_mark(%s, %s)", components, parameter)
        components = mark.add_mark(components, parameter)

        # Handle uơ in "huơ", "thuở", "quở"
        # If the current word has no last consonant and the first consonant
        # is one of "h", "th" and the vowel is "ươ" then change the vowel into
        # "uơ", keeping case and accent. If an alphabet character is then added
        # into the word then change back to "ươ".
        #
        # NOTE: In the dictionary, these are the only words having this strange
        # vowel so we don't need to worry about other cases.
        if accent.remove_accent_string(components[1]).lower() == "ươ" and \
                not components[2] and components[0].lower() in ["", "h", "th", "kh"]:
            # Backup accents
            ac = accent.get_accent_string(components[1])
            components[1] = ("u", "U")[components[1][0].isupper()] + components[1][1]
            components = accent.add_accent(components, ac)

    elif action == _Action.ADD_CHAR:
        if trans[0] == "<":
            if not components[2]:
                # Only allow ư, ơ or ươ sitting alone in the middle part
                # and ['g', 'i', '']. If we want to type giowf = 'giờ', separate()
                # will create ['g', 'i', '']. Therefore we have to allow
                # components[1] == 'i'.
                if (components[0].lower(), components[1].lower()) == ('g', 'i'):
                    components[0] += components[1]
                    components[1] = ''
                if not components[1] or \
                        (components[1].lower(), trans[1].lower()) == ('ư', 'ơ'):
                    components[1] += trans[1]
        else:
            components = utils.append_comps(components, parameter)
            if parameter.isalpha() and \
                    accent.remove_accent_string(components[1]).lower().startswith("uơ"):
                ac = accent.get_accent_string(components[1])
                components[1] = ('ư',  'Ư')[components[1][0].isupper()] + \
                    ('ơ', 'Ơ')[components[1][1].isupper()] + components[1][2:]
                components = accent.add_accent(components, ac)
    elif action == _Action.UNDO:
        components = _reverse(components, trans[1:])

    if action == _Action.ADD_MARK or (action == _Action.ADD_CHAR and parameter.isalpha()):
        # If there is any accent, remove and reapply it
        # because it is likely to be misplaced in previous transformations
        ac = accent.get_accent_string(components[1])

        if ac != accent.Accent.NONE:
            components = accent.add_accent(components, Accent.NONE)
            components = accent.add_accent(components, ac)

    logging.debug("After transform: %s", components)
    return components