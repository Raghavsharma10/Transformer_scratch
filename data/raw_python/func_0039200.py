def get_realnames(packages):
    """
    Return list of unique case-correct package names.

    Packages are listed in a case-insensitive sorted order.
    """
    return sorted({get_distribution(p).project_name for p in packages},
                  key=lambda n: n.lower())