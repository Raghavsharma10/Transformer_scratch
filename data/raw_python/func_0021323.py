def integer(prompt=None, empty=False):
    """Prompt an integer.

    Parameters
    ----------
    prompt : str, optional
        Use an alternative prompt.
    empty : bool, optional
        Allow an empty response.

    Returns
    -------
    int or None
        An int if the user entered a valid integer.
        None if the user pressed only Enter and ``empty`` was True.

    """
    s = _prompt_input(prompt)
    if empty and not s:
        return None
    else:
        try:
            return int(s)
        except ValueError:
            return integer(prompt=prompt, empty=empty)