def match(pattern):
    """
    Validates that a field value matches the regex given to this validator.
    """
    regex = re.compile(pattern)

    def validate(value):
        if not regex.match(value):
            return e("{} does not match the pattern {}", value, pattern)
    return validate