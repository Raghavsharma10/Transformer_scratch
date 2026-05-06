def infer(examples, alt_rules=None):
    """
    Returns a datetime.strptime-compliant format string for parsing the *most likely* date format
    used in examples. examples is a list containing example date strings.
    """
    date_classes = _tag_most_likely(examples)

    if alt_rules:
        date_classes = _apply_rewrites(date_classes, alt_rules)
    else:
        date_classes = _apply_rewrites(date_classes, RULES)

    date_string = ''
    for date_class in date_classes:
        date_string += date_class.directive

    return date_string