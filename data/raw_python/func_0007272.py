def positioning_headlines(headlines):
    """
    Strips unnecessary whitespaces/tabs if first header is not left-aligned
    """
    left_just = False
    for row in headlines:
        if row[-1] == 1:
            left_just = True
            break
    if not left_just:
        for row in headlines:
            row[-1] -= 1
    return headlines