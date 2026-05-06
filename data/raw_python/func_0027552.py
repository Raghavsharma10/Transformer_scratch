def is_in(allowed_values  # type: Set
          ):
    """
    'Values in' validation_function generator.
    Returns a validation_function to check that x is in the provided set of allowed values

    :param allowed_values: a set of allowed values
    :return:
    """
    def is_in_allowed_values(x):
        if x in allowed_values:
            return True
        else:
            # raise Failure('is_in: x in ' + str(allowed_values) + ' does not hold for x=' + str(x))
            raise NotInAllowedValues(wrong_value=x, allowed_values=allowed_values)

    is_in_allowed_values.__name__ = 'is_in_{}'.format(allowed_values)
    return is_in_allowed_values