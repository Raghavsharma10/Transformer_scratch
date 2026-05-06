def print_user_sets(wordsets, print_terms):
    """Print all user sets by title. If 'print_terms', also prints all terms of all user sets.
    :param wordsets: List of WordSet.
    :param print_terms: If True, also prints all terms of all user sets.
    """
    if not wordsets:
        print('No sets found')
    else:
        print('Found sets: {}'.format(len(wordsets)))
        for wordset in wordsets:
            print('    {}'.format(wordset))
            if print_terms:
                for term in wordset.terms:
                    print('        {}'.format(term))