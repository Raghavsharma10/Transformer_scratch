def find_recurring(number, min_repeat=5):
    """
    Attempts to find repeating digits in the fractional component of a number.

    Args:
        number(tuple): the number to process in the form:
            (int, int, int, ... ".", ... , int int int)
        min_repeat(int): the minimum number of times a pattern must occur to be
            defined as recurring. A min_repeat of n would mean a pattern must
            occur at least n + 1 times, so as to be repeated n times.

    Returns:
        The original number with repeating digits (if found) enclosed by  "["
        and "]" (tuple).

    Examples:
        >>> find_recurring((3, 2, 1, '.', 1, 2, 3, 1, 2, 3), min_repeat=1)
        (3, 2, 1, '.', '[', 1, 2, 3, ']')
    """
    # Return number if it has no fractional part, or min_repeat value invalid.
    if "." not in number or min_repeat < 1:
        return number
    # Seperate the number into integer and fractional parts.
    integer_part, fractional_part = integer_fractional_parts(number)
    # Reverse fractional part to get a sequence.
    sequence = fractional_part[::-1]
    # Initialize counters
    # The 'period' is the number of digits in a pattern.
    period = 0
    # The best pattern found will be stored.
    best = 0
    best_period = 0
    best_repeat = 0
    # Find recurring pattern.
    while (period < len(sequence)):
        period += 1
        pattern = sequence[:period]
        repeat = 0
        digit = period
        pattern_match = True
        while(pattern_match and digit < len(sequence)):
            for i, pattern_digit in enumerate(pattern):
                if sequence[digit + i] != pattern_digit:
                    pattern_match = False
                    break
            else:
                repeat += 1
            digit += period
        # Give each pattern found a rank and use the best.
        rank = period * repeat
        if rank > best:
            best_period = period
            best_repeat = repeat
            best = rank
    # If the pattern does not repeat often enough, return the original number.
    if best_repeat < min_repeat:
        return number
    # Use the best pattern found.
    pattern = sequence[:best_period]
    # Remove the pattern from our original number.
    number = integer_part + fractional_part[:-(best + best_period)]
    # Ensure we are at the start of the pattern.
    pattern_temp = pattern
    for i, digit in enumerate(pattern):
        if number[-1] == digit:
                number = number[:-1]
                pattern_temp = pattern_temp[1:] + (pattern_temp[0],)
    pattern = pattern_temp
    # Return the number with the recurring pattern enclosed with '[' and ']'.
    return number + ("[",) + pattern[::-1] + ("]",)