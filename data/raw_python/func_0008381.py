def predicative(adjective):
    """ Returns the predicative adjective (lowercase).
        In German, the attributive form preceding a noun is always used:
        "ein kleiner Junge" => strong, masculine, nominative,
        "eine schöne Frau" => mixed, feminine, nominative,
        "der kleine Prinz" => weak, masculine, nominative, etc.
        The predicative is useful for lemmatization.
    """
    w = adjective.lower()
    if len(w) > 3:
        for suffix in ("em", "en", "er", "es", "e"):
            if w.endswith(suffix):
                b = w[:max(-len(suffix), -(len(w)-3))]
                if b.endswith("bl"): # plausibles => plausibel
                    b = b[:-1] + "el"
                if b.endswith("pr"): # propres => proper
                    b = b[:-1] + "er"
                return b
    return w