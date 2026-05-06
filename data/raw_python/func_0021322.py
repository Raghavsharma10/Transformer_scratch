def email(prompt=None, empty=False, mode="simple"):
    """Prompt an email address.

    This check is based on a simple regular expression and does not verify
    whether an email actually exists.

    Parameters
    ----------
    prompt : str, optional
        Use an alternative prompt.
    empty : bool, optional
        Allow an empty response.
    mode : {'simple'}, optional
        'simple' will use a simple regular expression.
        No other mode is implemented yet.

    Returns
    -------
    str or None
        A str if the user entered a likely email address.
        None if the user pressed only Enter and ``empty`` was True.

    """
    if mode == "simple":
        s = _prompt_input(prompt)
        if empty and not s:
            return None
        else:
            if RE_EMAIL_SIMPLE.match(s):
                return s
            else:
                return email(prompt=prompt, empty=empty, mode=mode)
    else:
        raise ValueError