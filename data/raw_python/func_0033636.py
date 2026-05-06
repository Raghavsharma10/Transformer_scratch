def get_telex_definition(w_shorthand=True, brackets_shorthand=True):
    """Create a definition dictionary for the TELEX input method

    Args:
        w_shorthand (optional): allow a stand-alone w to be
            interpreted as an ư. Default to True.
        brackets_shorthand (optional, True): allow typing ][ as
            shorthand for ươ. Default to True.

    Returns a dictionary to be passed into process_key().
    """
    telex = {
        "a": "a^",
        "o": "o^",
        "e": "e^",
        "w": ["u*", "o*", "a+"],
        "d": "d-",
        "f": "\\",
        "s": "/",
        "r": "?",
        "x": "~",
        "j": ".",
    }

    if w_shorthand:
        telex["w"].append('<ư')

    if brackets_shorthand:
        telex.update({
            "]": "<ư",
            "[": "<ơ",
            "}": "<Ư",
            "{": "<Ơ"
        })

    return telex