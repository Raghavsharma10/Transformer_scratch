def get_common_terms(*api_envs):
    """Get all term duplicates across all user word sets as a list of
    (title of first word set, title of second word set, set of terms) tuples."""
    common_terms = []
    # pylint: disable=no-value-for-parameter
    wordsets = get_user_sets(*api_envs)
    # pylint: enable=no-value-for-parameter

    for wordset1, wordset2 in combinations(wordsets, 2):
        common = wordset1.has_common(wordset2)
        if common:
            common_terms.append((wordset1.title, wordset2.title, common))
    return common_terms