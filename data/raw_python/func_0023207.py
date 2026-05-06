def next_power_of_2(n):
    """ Return next power of 2 greater than or equal to n """
    n -= 1  # greater than OR EQUAL TO n
    shift = 1
    while (n + 1) & n:  # n+1 is not a power of 2 yet
        n |= n >> shift
        shift *= 2
    return max(4, n + 1)