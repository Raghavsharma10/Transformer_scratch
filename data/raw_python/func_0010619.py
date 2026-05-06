def _style(enabled, text, **kwargs):
    """ Helper function to enable/disable styled output text.

    Args:
        enable (bool): Turn on or off styling.
        text (string): The string that should be styled.
        kwargs (dict): Parameters that are passed through to click.style

    Returns:
        string: The input with either the styling applied (enabled=True)
                or just the text (enabled=False)
    """
    if enabled:
        return click.style(text, **kwargs)
    else:
        return text