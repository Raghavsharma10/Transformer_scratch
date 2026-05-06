def validate_available_choice(enum, to_value):
    """
    Validate that to_value is defined as a value in enum.
    """
    if to_value is None:
        return

    if type(to_value) is not int:
        try:
            to_value = int(to_value)
        except ValueError:
            message_str = "'{value}' cannot be converted to int"
            message = _(six.text_type(message_str))
            raise InvalidStatusOperationError(message.format(value=to_value))

    if to_value not in list(dict(enum.choices()).keys()):
        message = _(six.text_type('Select a valid choice. {value} is not one of the available choices.'))
        raise InvalidStatusOperationError(message.format(value=to_value))