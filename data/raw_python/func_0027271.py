def get_validators_description(view):
    """
    Returns validators description in format:
    ### Validators:
    * validator1 name
     * validator1 docstring
    * validator2 name
     * validator2 docstring
    """
    action = getattr(view, 'action', None)
    if action is None:
        return ''

    description = ''
    validators = getattr(view, action + '_validators', [])
    for validator in validators:
        validator_description = get_entity_description(validator)
        description += '\n' + validator_description if description else validator_description

    return '### Validators:\n' + description if description else ''