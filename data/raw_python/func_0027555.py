def is_superset(reference_set  # type: Set
                ):
    """
    'Is superset' validation_function generator.
    Returns a validation_function to check that x is a superset of reference_set

    :param reference_set: the reference set
    :return:
    """
    def is_superset_of(x):
        missing = reference_set - x
        if len(missing) == 0:
            return True
        else:
            # raise Failure('is_superset: len(reference_set - x) == 0 does not hold for x=' + str(x)
            #               + ' and reference_set=' + str(reference_set) + '. x does not contain required '
            #                       'elements ' + str(missing))
            raise NotSuperset(wrong_value=x, reference_set=reference_set, missing=missing)

    is_superset_of.__name__ = 'is_superset_of_{}'.format(reference_set)
    return is_superset_of