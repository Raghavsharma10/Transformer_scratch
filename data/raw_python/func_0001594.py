def between(min_value, max_value):
    """
    Validates that a field value is between the two values
    given to this validator.
    """
    def validate(value):
        if value < min_value:
            return e("{} is not greater than or equal to {}",
                value, min_value)
        if value > max_value:
            return e("{} is not less than or equal to {}",
                value, max_value)
    return validate