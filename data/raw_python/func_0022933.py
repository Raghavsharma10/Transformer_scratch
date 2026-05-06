def _make_patterns(patterns):
    """Create a ScreenPatternList from a given pattern text.

    Args:
        pattern_txt (str list): the patterns

    Returns:
        mpdlcd.display_pattern.ScreenPatternList: a list of patterns from the
            given entries.
    """
    field_registry = display_fields.FieldRegistry()

    pattern_list = display_pattern.ScreenPatternList(
        field_registry=field_registry,
    )
    for pattern in patterns:
        pattern_list.add(pattern.split('\n'))
    return pattern_list