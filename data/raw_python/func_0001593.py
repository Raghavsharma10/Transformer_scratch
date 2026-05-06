def lt(lt_value):
    """
    Validates that a field value is less than the
    value given to this validator.
    """
    def validate(value):
        if value >= lt_value:
            return e("{} is not less than {}", value, lt_value)
    return validate