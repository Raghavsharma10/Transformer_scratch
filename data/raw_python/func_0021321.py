def character(prompt=None, empty=False):
    """Prompt a single character.

    Parameters
    ----------
    prompt : str, optional
        Use an alternative prompt.
    empty : bool, optional
        Allow an empty response.

    Returns
    -------
    str or None
        A str if the user entered a single-character, non-empty string.
        None if the user pressed only Enter and ``empty`` was True.

    """
    s = _prompt_input(prompt)
    if empty and not s:
        return None
    elif len(s) == 1:
        return s
    else:
        return character(prompt=prompt, empty=empty)