def string(prompt=None, empty=False):
    """Prompt a string.

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

    """
    s = _prompt_input(prompt)
    if empty and not s:
        return None
    else:
        if s:
            return s
        else:
            return string(prompt=prompt, empty=empty)