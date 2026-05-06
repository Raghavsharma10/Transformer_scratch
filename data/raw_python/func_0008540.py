def get_credits():
    """Extract credits from `AUTHORS.rst`"""
    credits = read(os.path.join(_HERE, "AUTHORS.rst")).split("\n")
    from_index = credits.index("Active Contributors")
    credits = "\n".join(credits[from_index + 2:])
    return credits