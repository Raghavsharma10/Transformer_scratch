def _append(lst, indices, value):
    """Adds `value` to `lst` list indexed by `indices`. Will create sub lists as required.
    """
    for i, idx in enumerate(indices):
        # We need to loop because sometimes indices can increment by more than 1 due to missing tokens.
        # Example: Sentence with no words after filtering words.
        while len(lst) <= idx:
            # Update max counts whenever a new sublist is created.
            # There is no need to worry about indices beyond `i` since they will end up creating new lists as well.
            lst.append([])
        lst = lst[idx]

    # Add token and update token max count.
    lst.append(value)