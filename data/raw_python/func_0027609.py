def is_multiple_of(ref):
    """ Validates that x is a multiple of the reference (`x % ref == 0`) """
    def is_multiple_of_ref(x):
        if x % ref == 0:
            return True
        else:
            raise IsNotMultipleOf(wrong_value=x, ref=ref)
            # raise Failure('x % {ref} == 0 does not hold for x={val}'.format(ref=ref, val=x))

    is_multiple_of_ref.__name__ = 'is_multiple_of_{}'.format(ref)
    return is_multiple_of_ref