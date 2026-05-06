def validate_options(options):
    """
    Validate the options that course author set up and return errors in a dict if there is any
    """
    errors = []

    if int(options['rationale_size']['min']) < 1:
        errors.append(_('Minimum Characters'))
    if int(options['rationale_size']['max']) < 0 or int(options['rationale_size']['max']) > MAX_RATIONALE_SIZE:
        errors.append(_('Maximum Characters'))
    if not any(error in [_('Minimum Characters'), _('Maximum Characters')] for error in errors) \
            and int(options['rationale_size']['max']) <= int(options['rationale_size']['min']):
        errors += [_('Minimum Characters'), _('Maximum Characters')]
    try:
        if options['algo']['num_responses'] != '#' and int(options['algo']['num_responses']) < 0:
            errors.append(_('Number of Responses'))
    except ValueError:
        errors.append(_('Not an Integer'))

    if not errors:
        return None
    else:
        return {'options_error': _('Invalid Option(s): ') + ', '.join(errors)}