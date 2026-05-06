def _guess_name(desc, taken=None):
    """Attempts to guess the menu entry name from the function name."""
    taken = taken or []
    name = ""
    # Try to find the shortest name based on the given description.
    for word in desc.split():
        c = word[0].lower()
        if not c.isalnum():
            continue
        name += c
        if name not in taken:
            break
    # If name is still taken, add a number postfix.
    count = 2
    while name in taken:
        name = name + str(count)
        count += 1
    return name