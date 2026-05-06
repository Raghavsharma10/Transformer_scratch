def indent(txt, spacing=4):
    """
    Indent given text using custom spacing, default is set to 4.
    """
    return prefix(str(txt), ''.join([' ' for _ in range(spacing)]))