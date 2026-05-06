def print_common_terms(common_terms):
    """Print common terms for each pair of word sets.
    :param common_terms: Output of get_common_terms().
    """
    if not common_terms:
        print('No duplicates')
    else:
        for set_pair in common_terms:
            set1, set2, terms = set_pair
            print('{} and {} have in common:'.format(set1, set2))
            for term in terms:
                print('    {}'.format(term))