def equate_nickname(name1, name2):
    """
    Evaluates whether names match based on common nickname patterns

    This is not currently used in any name comparison
    """
    # Convert '-ie' and '-y' to the root name
    nickname_regex = r'(.)\1(y|ie)$'
    root_regex = r'\1'

    name1 = re.sub(nickname_regex, root_regex, name1)
    name2 = re.sub(nickname_regex, root_regex, name2)

    if equate_prefix(name1, name2):
        return True

    return False