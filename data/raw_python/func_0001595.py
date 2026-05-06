def length(min=None, max=None):
    """
    Validates that a field value's length is between the bounds given to this
    validator.
    """

    def validate(value):
        if min and len(value) < min:
            return e("{} does not have a length of at least {}", value, min)
        if max and len(value) > max:
            return e("{} does not have a length of at most {}", value, max)
    return validate