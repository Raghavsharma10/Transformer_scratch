def validate_seeded_answers_simple(answers, options, algo):
    """
    This validator checks if the answers includes all possible options

    Args:
        answers (str): the answers to be checked
        options (dict): all options that should exist in the answers
        algo (str): selection algorithm

    Returns:
        None if everything is good. Otherwise, the missing option error message.
    """
    seen_options = {}
    for answer in answers:
        if answer:
            key = options[answer['answer']].get('text')
            if options[answer['answer']].get('image_url'):
                key += options[answer['answer']].get('image_url')
            seen_options.setdefault(key, 0)
            seen_options[key] += 1

    missing_options = []
    index = 1
    for option in options:
        key = option.get('text') + option.get('image_url') if option.get('image_url') else option.get('text')
        if option.get('text') != 'n/a':
            if seen_options.get(key, 0) == 0:
                missing_options.append(_('Option ') + str(index))
            index += 1

    if missing_options:
        return {'seed_error': _('Missing option seed(s): ') + ', '.join(missing_options)}

    return None