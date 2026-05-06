def build_path(entities, path_patterns, strict=False):
    """
    Constructs a path given a set of entities and a list of potential
    filename patterns to use.

    Args:
        entities (dict): A dictionary mapping entity names to entity values.
        path_patterns (str, list): One or more filename patterns to write
            the file to. Entities should be represented by the name
            surrounded by curly braces. Optional portions of the patterns
            should be denoted by square brackets. Entities that require a
            specific value for the pattern to match can pass them inside
            carets. Default values can be assigned by specifying a string after
            the pipe operator. E.g., (e.g., {type<image>|bold} would only match
            the pattern if the entity 'type' was passed and its value is
            "image", otherwise the default value "bold" will be used).
                Example 1: 'sub-{subject}/[var-{name}/]{id}.csv'
                Result 2: 'sub-01/var-SES/1045.csv'
        strict (bool): If True, all passed entities must be matched inside a
            pattern in order to be a valid match. If False, extra entities will
            be ignored so long as all mandatory entities are found.

    Returns:
        A constructed path for this file based on the provided patterns.
    """
    if isinstance(path_patterns, string_types):
        path_patterns = [path_patterns]

    # Loop over available patherns, return first one that matches all
    for pattern in path_patterns:
        # If strict, all entities must be contained in the pattern
        if strict:
            defined = re.findall('\{(.*?)(?:<[^>]+>)?\}', pattern)
            if set(entities.keys()) - set(defined):
                continue
        # Iterate through the provided path patterns
        new_path = pattern
        optional_patterns = re.findall('\[(.*?)\]', pattern)
        # First build from optional patterns if possible
        for optional_pattern in optional_patterns:
            optional_chunk = replace_entities(entities, optional_pattern) or ''
            new_path = new_path.replace('[%s]' % optional_pattern,
                                        optional_chunk)
        # Replace remaining entities
        new_path = replace_entities(entities, new_path)

        if new_path:
            return new_path

    return None