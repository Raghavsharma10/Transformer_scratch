def lte(max_value):
    """
    Validates that a field value is less than or equal to the
    value given to this validator.
    """
    def validate(value):
        if value > max_value:
            return e("{} is not less than or equal to {}", value, max_value)
    return validate