def average(numbers, averagetype='mean'):
    """
    Find the average of a list of numbers

    :type numbers: list
    :param numbers: The list of numbers to find the average of.

    :type averagetype: string
    :param averagetype: The type of average to find.

    >>> average([1, 2, 3, 4, 5], 'median')
    3
    """

    try:
        # Try to get the mean of the numbers
        statistics.mean(numbers)

    except RuntimeError:
        # Raise a warning
        raise ValueError('Unable to parse the list.')

    # If the lowercase version of the average type is 'mean'
    if averagetype.lower() == 'mean':
        # Return the answer
        return statistics.mean(numbers)

    # If the lowercase version of the average type is 'mode'
    elif averagetype.lower() == 'mode':
        # Return the answer
        return statistics.mode(numbers)

    # If the lowercase version of the average type is 'median'
    elif averagetype.lower() == 'median':
        # Return the answer
        return statistics.median(numbers)

    # If the lowercase version of the average type is 'min'
    elif averagetype.lower() == 'min':
        # Return the answer
        return min(numbers)

    # If the lowercase version of the average type is 'max'
    elif averagetype.lower() == 'max':
        # Return the answer
        return max(numbers)

    # If the lowercase version of the average type is 'range'
    elif averagetype.lower() == 'range':
        # Return the answer
        return max(numbers) - min(numbers)

    # Raise a warning
    raise ValueError('Invalid average type provided.')