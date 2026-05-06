def _percent_match(date_classes, tokens):
    """
    For each date class, return the percentage of tokens that the class matched (floating point [0.0 - 1.0]). The
    returned value is a tuple of length patterns. Tokens should be a list.
    """
    match_count = [0] * len(date_classes)

    for i, date_class in enumerate(date_classes):
        for token in tokens:
            if date_class.is_match(token):
                match_count[i] += 1

    percentages = tuple([float(m) / len(tokens) for m in match_count])
    return percentages