def gen_anytext(*args):
    """
    Convenience function to create bag of words for anytext property
    """

    bag = []

    for term in args:
        if term is not None:
            if isinstance(term, list):
                for term2 in term:
                    if term2 is not None:
                        bag.append(term2)
            else:
                bag.append(term)
    return ' '.join(bag)