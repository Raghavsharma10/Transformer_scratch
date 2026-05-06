def textalign(text, maxlength, align='left'):
    """
    Align Text When Given Full Length
    """
    if align == 'left':
        return text
    elif align == 'centre' or align == 'center':
        spaces = ' ' * (int((maxlength - len(text)) / 2))
    elif align == 'right':
        spaces = (maxlength - len(text))
    else:
        raise ValueError("Invalid alignment specified.")
    return spaces + text