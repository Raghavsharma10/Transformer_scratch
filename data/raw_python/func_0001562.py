def _suffixArrayWithTrace(s, SA, n, K, operations, totalOperations):
    """
    This function is a rewrite in Python of the C implementation proposed in Kärkkäinen and Sanders paper.

    Find the suffix array SA of s[0..n-1] in {1..K}^n
    Require s[n]=s[n+1]=s[n+2]=0, n>=2
    """
    if _trace:
        _traceSuffixArray(operations, totalOperations)

    n0 = (n + 2) // 3
    n1 = (n + 1) // 3
    n2 = n // 3
    n02 = n0 + n2

    SA12 = _array("i", [0] * (n02 + 3))
    SA0 = _array("i", [0] * n0)
    s0 = _array("i", [0] * n0)

    # s12 : positions of mod 1 and mod 2 suffixes
    s12 = _array("i", [i for i in range(n + (n0 - n1)) if i % 3])  # <- writing i%3 is more efficient than i%3!=0
    s12.extend([0] * 3)

    # lsb radix sort the mod 1 and mod 2 triples
    _radixPass(s12, SA12, s[2:], n02, K)
    if _trace:
        operations += n02
        _traceSuffixArray(operations, totalOperations)

    _radixPass(SA12, s12, s[1:], n02, K)
    if _trace:
        operations += n02
        _traceSuffixArray(operations, totalOperations)

    _radixPass(s12, SA12, s, n02, K)
    if _trace:
        operations += n02
        _traceSuffixArray(operations, totalOperations)

    # find lexicographic names of triples
    name = 0
    c = _array("i", [-1] * 3)
    for i in range(n02):
        cSA12 = s[SA12[i]:SA12[i] + 3]
        if cSA12 != c:
            name += 1
            c = cSA12

        if SA12[i] % 3 == 1:
            s12[SA12[i] // 3] = name  # left half
        else:
            s12[(SA12[i] // 3) + n0] = name  # right half

    if name < n02:  # recurse if names are not yet unique
        operations = _suffixArrayWithTrace(s12, SA12, n02, name + 1, operations, totalOperations)
        if _trace:
            _traceSuffixArray(operations, totalOperations)

        # store unique names in s12 using the suffix array
        for i, SA12_i in enumerate(SA12[:n02]):
            s12[SA12_i] = i + 1
    else:  # generate the suffix array of s12 directly
        if _trace:
            operations += _nbOperations(n02)
            _traceSuffixArray(operations, totalOperations)

        for i, s12_i in enumerate(s12[:n02]):
            SA12[s12_i - 1] = i

    # stably sort the mod 0 suffixes from SA12 by their first character
    j = 0
    for SA12_i in SA12[:n02]:
        if (SA12_i < n0):
            s0[j] = 3 * SA12_i
            j += 1

    _radixPass(s0, SA0, s, n0, K)
    if _trace:
        operations += n0
        _traceSuffixArray(operations, totalOperations)

    # merge sorted SA0 suffixes and sorted SA12 suffixes
    p = j = k = 0
    t = n0 - n1
    while k < n:
        if SA12[t] < n0:  # pos of current offset 12 suffix
            i = SA12[t] * 3 + 1
        else:
            i = (SA12[t] - n0) * 3 + 2

        j = SA0[p]  # pos of current offset 0 suffix

        if SA12[t] < n0:
            bool = (s[i], s12[SA12[t] + n0]) <= (s[j], s12[int(j / 3)])
        else:
            bool = (s[i], s[i + 1], s12[SA12[t] - n0 + 1]) <= (s[j], s[j + 1], s12[int(j / 3) + n0])

        if (bool):
            SA[k] = i
            t += 1
            if t == n02:  # done --- only SA0 suffixes left
                k += 1
                while p < n0:
                    SA[k] = SA0[p]
                    p += 1
                    k += 1

        else:
            SA[k] = j
            p += 1
            if p == n0:  # done --- only SA12 suffixes left
                k += 1
                while t < n02:
                    if SA12[t] < n0:  # pos of current offset 12 suffix
                        SA[k] = (SA12[t] * 3) + 1
                    else:
                        SA[k] = ((SA12[t] - n0) * 3) + 2
                    t += 1
                    k += 1
        k += 1
    return operations