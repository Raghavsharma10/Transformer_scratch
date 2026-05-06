def distinct():
    """
    Validates that all items in the given field list value are distinct,
    i.e. that the list contains no duplicates.
    """
    def validate(value):
        for i, item in enumerate(value):
            if item in value[i+1:]:
                return e("{} is not a distinct set of values", value)
    return validate