def mutateString(original, n, replacements='acgt'):
    """
    Mutate C{original} in C{n} places with chars chosen from C{replacements}.

    @param original: The original C{str} to mutate.
    @param n: The C{int} number of locations to mutate.
    @param replacements: The C{str} of replacement letters.

    @return: A new C{str} with C{n} places of C{original} mutated.
    @raises ValueError: if C{n} is too high, or C{replacement} contains
        duplicates, or if no replacement can be made at a certain locus
        because C{replacements} is of length one, or if C{original} is of
        zero length.
    """
    if not original:
        raise ValueError('Empty original string passed.')

    if n > len(original):
        raise ValueError('Cannot make %d mutations in a string of length %d' %
                         (n, len(original)))

    if len(replacements) != len(set(replacements)):
        raise ValueError('Replacement string contains duplicates')

    if len(replacements) == 1 and original.find(replacements) != -1:
        raise ValueError('Impossible replacement')

    result = list(original)
    length = len(original)

    for offset in range(length):
        if uniform(0.0, 1.0) < float(n) / (length - offset):
            # Mutate.
            while True:
                new = choice(replacements)
                if new != result[offset]:
                    result[offset] = new
                    break
            n -= 1
            if n == 0:
                break

    return ''.join(result)