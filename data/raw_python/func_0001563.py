def _longestCommonPrefix(seq1, seq2, start1=0, start2=0):
    """
    Returns the length of the longest common prefix of seq1
    starting at offset start1 and seq2 starting at offset start2.

    >>> _longestCommonPrefix("abcdef", "abcghj")
    3

    >>> _longestCommonPrefix("abcghj", "abcdef")
    3

    >>> _longestCommonPrefix("miss", "")
    0

    >>> _longestCommonPrefix("", "mr")
    0

    >>> _longestCommonPrefix(range(128), range(128))
    128

    >>> _longestCommonPrefix("abcabcabc", "abcdefabcdef", 0, 6)
    3

    >>> _longestCommonPrefix("abcdefabcdef", "abcabcabc", 6, 0)
    3

    >>> _longestCommonPrefix("abc", "abcabc", 1, 4)
    2

    >>> _longestCommonPrefix("abcabc", "abc", 4, 1)
    2
    """

    len1 = len(seq1) - start1
    len2 = len(seq2) - start2

    # We set seq2 as the shortest sequence
    if len1 < len2:
        seq1, seq2 = seq2, seq1
        start1, start2 = start2, start1
        len1, len2 = len2, len1

    # if seq2 is empty returns 0
    if len2 == 0:
        return 0

    i = 0
    pos2 = start2
    for i in range(min(len1, len2)):
        # print seq1, seq2, start1, start2
        if seq1[start1 + i] != seq2[start2 + i]:
            return i

    # we have reached the end of seq2 (need to increment i)
    return i + 1