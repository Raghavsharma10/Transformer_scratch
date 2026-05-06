def real(prompt=None, empty=False):
    """Prompt a real number.

    Parameters
    ----------
    prompt : str, optional
        Use an alternative prompt.
    empty : bool, optional
        Allow an empty response.

    Returns
    -------
    float or None
        A float if the user entered a valid real number.
        None if the user pressed only Enter and ``empty`` was True.

    """
    s = _prompt_input(prompt)
    if empty and not s:
        return None
    else:
        try:
            return float(s)
        except ValueError:
            return real(prompt=prompt, empty=empty)