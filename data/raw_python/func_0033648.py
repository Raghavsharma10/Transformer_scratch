def is_valid_mark(comps, mark_trans):
    """
    Check whether the mark given by mark_trans is valid to add to the components
    """
    if mark_trans == "*_":
        return True
    components = list(comps)

    if mark_trans[0] == 'd' and components[0] \
            and components[0][-1].lower() in ("d", "đ"):
        return True
    elif components[1] != "" and \
            strip(components[1]).lower().find(mark_trans[0]) != -1:
        return True
    else:
        return False