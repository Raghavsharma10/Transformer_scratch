def sequenceCategoryLengths(read, categories, defaultCategory=None,
                            suppressedCategory='...', minLength=1):
    """
    Summarize the nucleotides or AAs found in a read by assigning each to a
    category and reporting the lengths of the contiguous category classes
    found along the sequence.

    @param read: A C{Read} instance or one of its subclasses.
    @param categories: A C{dict} mapping nucleotides or AAs to category.
    @param defaultCategory: The category to use if a sequence base is not
        in C{categories}.
    @param suppressedCategory: The category to use to indicate suppressed
        sequence regions (i.e., made up of stretches of bases that are less
        than C{minLength} in length).
    @param minLength: stretches of the read that are less than this C{int}
        length will be summed and reported as being in the
        C{suppressedCategory} category.
    @raise ValueError: If minLength is less than one.
    @return: A C{list} of 2-C{tuples}. Each tuple contains a (category, count).
    """
    result = []
    append = result.append
    get = categories.get
    first = True
    currentCategory = None
    currentCount = 0
    suppressing = False
    suppressedCount = 0

    if minLength < 1:
        raise ValueError('minLength must be at least 1')

    for base in read.sequence:
        thisCategory = get(base, defaultCategory)
        if first:
            first = False
            currentCategory = thisCategory
            currentCount += 1
        else:
            if thisCategory == currentCategory:
                # This base is still in the same category as the last base.
                # Keep counting.
                currentCount += 1
            else:
                # This is a new category.
                if currentCount < minLength:
                    # The category region that was just seen will not be
                    # emitted.
                    if suppressing:
                        # Already suppressing. Suppress the just-seen
                        # region too.
                        suppressedCount += currentCount
                    else:
                        # Start suppressing.
                        suppressedCount = currentCount
                        suppressing = True
                else:
                    if suppressing:
                        append((suppressedCategory, suppressedCount))
                        suppressedCount = 0
                        suppressing = False
                    append((currentCategory, currentCount))
                currentCategory = thisCategory
                currentCount = 1

    if suppressing:
        append((suppressedCategory, suppressedCount + currentCount))
    elif currentCount >= minLength:
        append((currentCategory, currentCount))
    elif currentCount:
        append((suppressedCategory, currentCount))

    return result