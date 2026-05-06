def finalizeSection(section):
    """
    Correct the nxt and parent for each child
    """
    cur = section.first_child
    last = section.last_child
    if last is not None:
        last.nxt = None

    while cur is not None:
        cur.parent = section
        cur = cur.nxt