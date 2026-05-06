def acceptable(value, capitalize=False):
    """Convert a string into something that can be used as a valid python variable name"""
    name = regexes['punctuation'].sub("", regexes['joins'].sub("_", value))
    # Clean up irregularities in underscores.
    name = regexes['repeated_underscore'].sub("_", name.strip('_'))
    if capitalize:
        # We don't use python's built in capitalize method here because it
        # turns all upper chars into lower chars if not at the start of
        # the string and we only want to change the first character.
        name_parts = []
        for word in name.split('_'):
            name_parts.append(word[0].upper())
            if len(word) > 1:
                name_parts.append(word[1:])
        name = ''.join(name_parts)
    return name