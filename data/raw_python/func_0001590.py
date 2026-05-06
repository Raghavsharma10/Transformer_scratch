def gte(min_value):
    """
    Validates that a field value is greater than or equal to the
    value given to this validator.
    """
    def validate(value):
        if value < min_value:
            return e("{} is not greater than or equal to {}", value, min_value)
    return validate