def confirm(text, default=True):
    """
    Console confirmation dialog based on raw_input.
    """
    if default:
        legend = "[y]/n"
    else:
        legend = "y/[n]"
    res = ""
    while (res != "y") and (res != "n"):
        res = raw_input(text + " ({}): ".format(legend)).lower()
        if not res and default:
            res = "y"
        elif not res and not default:
            res = "n"
    if res[0] == "y":
        return True
    else:
        return False