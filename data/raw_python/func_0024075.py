def generate_set_partition_strings(n):
    """Generate the restricted growth strings for all of the partitions of an `n`-member set.
    
    Uses Algorithm H from page 416 of volume 4A of Knuth's `The Art of Computer
    Programming`. Returns the partitions in lexicographical order.
    
    Parameters
    ----------
    n : scalar int, non-negative
        Number of (unique) elements in the set to be partitioned.
    
    Returns
    -------
    partitions : list of :py:class:`Array`
        List has a number of elements equal to the `n`-th Bell number (i.e.,
        the number of partitions for a set of size `n`). Each element has
        length `n`, the elements of which are the restricted growth strings
        describing the partitions of the set. The strings are returned in
        lexicographic order.
    """
    # Handle edge cases:
    if n == 0:
        return []
    elif n == 1:
        return [scipy.array([0])]
    
    partitions = []
    
    # Step 1: Initialize
    a = scipy.zeros(n, dtype=int)
    b = scipy.ones(n, dtype=int)
    
    while True:
        # Step 2: Visit
        partitions.append(a.copy())
        if a[-1] == b[-1]:
            # Step 4: Find j. j is the index of the first element from the end
            # for which a != b, with the exception of the last element.
            j = (a[:-1] != b[:-1]).nonzero()[0][-1]
            # Step 5: Increase a_j (or terminate):
            if j == 0:
                break
            else:
                a[j] += 1
                # Step 6: Zero out a_{j+1} to a_n:
                b[-1] = b[j] + (a[j] == b[j])
                a[j + 1:] = 0
                b[j + 1 :-1] = b[-1]
        else:
            # Step 3: Increase a_n:
            a[-1] += 1
    
    return partitions