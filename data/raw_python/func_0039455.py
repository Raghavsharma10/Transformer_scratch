def string_range(last):
    """Compute the range of string between "a" and last.

    This works for simple "a to z" lists, but also for "a to zz" lists.
    """
    for k in range(len(last)):
        for x in product(string.ascii_lowercase, repeat=k+1):
            result = ''.join(x)
            yield result
            if result == last:
                return