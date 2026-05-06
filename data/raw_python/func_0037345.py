def check_dependencies():
    """Check external dependecies
    Return a tuple with the available generators.
    """
    available = []
    try:
        shell('ebook-convert')
        available.append('calibre')
    except OSError:
        pass
    try:
        shell('pandoc --help')
        available.append('pandoc')
    except OSError:
        pass
    if not available:
        sys.exit(error('No generator found, you cannot use md2ebook.'))
    check_dependency_epubcheck()
    return available