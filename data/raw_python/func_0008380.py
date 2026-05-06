def attributive(adjective, gender=MALE, role=SUBJECT, article=None):
    """ For a predicative adjective, returns the attributive form (lowercase).
        In German, the attributive is formed with -e, -em, -en, -er or -es,
        depending on gender (masculine, feminine, neuter or plural) and role
        (nominative, accusative, dative, genitive).
    """
    w, g, c, a = \
        adjective.lower(), gender[:1].lower(), role[:3].lower(), article and article.lower() or None
    if w in adjective_attributive:
        return adjective_attributive[w]
    if a is None \
    or a in ("mir", "dir", "ihm") \
    or a in ("ein", "etwas", "mehr") \
    or a.startswith(("all", "mehrer", "wenig", "viel")):
        return w + adjectives_strong.get((g, c), "")
    if a.startswith(("ein", "kein")) \
    or a.startswith(("mein", "dein", "sein", "ihr", "Ihr", "unser", "euer")):
        return w + adjectives_mixed.get((g, c), "")
    if a in ("arm", "alt", "all", "der", "die", "das", "den", "dem", "des") \
    or a.startswith((
      "derselb", "derjenig", "jed", "jeglich", "jen", "manch", 
      "dies", "solch", "welch")):
        return w + adjectives_weak.get((g, c), "")
    # Default to strong inflection.
    return w + adjectives_strong.get((g, c), "")