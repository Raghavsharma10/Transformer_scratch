def secret(prompt=None, empty=False):
    """Prompt a string without echoing.

    Parameters
    ----------
    prompt : str, optional
        Use an alternative prompt.
    empty : bool, optional
        Allow an empty response.

    Returns
    -------
    str or None
        A str if the user entered a non-empty string.
        None if the user pressed only Enter and ``empty`` was True.

    Raises
    ------
    getpass.GetPassWarning
        If echo free input is unavailable.

    See Also
    --------
    getpass.getpass

    """
    if prompt is None:
        prompt = PROMPT
    s = getpass.getpass(prompt=prompt)
    if empty and not s:
        return None
    else:
        if s:
            return s
        else:
            return secret(prompt=prompt, empty=empty)