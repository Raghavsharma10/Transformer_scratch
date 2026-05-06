def regex(pattern, prompt=None, empty=False, flags=0):
    """Prompt a string that matches a regular expression.

    Parameters
    ----------
    pattern : str
        A regular expression that must be matched.
    prompt : str, optional
        Use an alternative prompt.
    empty : bool, optional
        Allow an empty response.
    flags : int, optional
        Flags that will be passed to ``re.match``.

    Returns
    -------
    Match or None
        A match object if the user entered a matching string.
        None if the user pressed only Enter and ``empty`` was True.

    See Also
    --------
    re.match

    """
    s = _prompt_input(prompt)
    if empty and not s:
        return None
    else:
        m = re.match(pattern, s, flags=flags)
        if m:
            return m
        else:
            return regex(pattern, prompt=prompt, empty=empty, flags=flags)