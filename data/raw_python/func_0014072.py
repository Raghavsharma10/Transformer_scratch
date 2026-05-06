def validate_valid_transition(enum, from_value, to_value):
    """
    Validate that to_value is a valid choice and that to_value is a valid transition from from_value.
    """
    validate_available_choice(enum, to_value)
    if hasattr(enum, '_transitions') and not enum.is_valid_transition(from_value, to_value):
        message = _(six.text_type('{enum} can not go from "{from_value}" to "{to_value}"'))
        raise InvalidStatusOperationError(message.format(
            enum=enum.__name__,
            from_value=enum.name(from_value),
            to_value=enum.name(to_value) or to_value
        ))