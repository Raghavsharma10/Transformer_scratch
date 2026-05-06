def lowstrip(term):
    """Convert to lowercase and strip spaces"""
    term = re.sub('\s+', ' ', term)
    term = term.lower()
    return term