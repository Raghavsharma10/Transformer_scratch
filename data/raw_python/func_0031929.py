def _handle_negatives(numbers):
    """
        Add the minimum negative number to all the numbers in the
        such that all the elements become >= 0
    """
    min_number = min(filter(lambda x : type(x)==int,numbers))
    if min_number < 0:
        return [x+abs(min_number) if type(x)==int else x for x in numbers]
    else:
        return numbers