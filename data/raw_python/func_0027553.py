def is_subset(reference_set  # type: Set
              ):
    """
    'Is subset' validation_function generator.
    Returns a validation_function to check that x is a subset of reference_set

    :param reference_set: the reference set
    :return:
    """
    def is_subset_of(x):
        missing = x - reference_set
        if len(missing) == 0:
            return True
        else:
            # raise Failure('is_subset: len(x - reference_set) == 0 does not hold for x=' + str(x)
            #                   + ' and reference_set=' + str(reference_set) + '. x contains unsupported '
            #                      'elements ' + str(missing))
            raise NotSubset(wrong_value=x, reference_set=reference_set, unsupported=missing)

    is_subset_of.__name__ = 'is_subset_of_{}'.format(reference_set)
    return is_subset_of