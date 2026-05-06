def _apply_rewrites(date_classes, rules):
    """
    Return a list of date elements by applying rewrites to the initial date element list
    """
    for rule in rules:
        date_classes = rule.execute(date_classes)

    return date_classes