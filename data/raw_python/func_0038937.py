def long_description():
    """Return the contents of README.rst (with badging removed)."""
    # use re.compile() for flags support in Python 2.6
    pattern = re.compile(r'\n^\.\. start-badges.*^\.\. end-badges\n',
                         flags=re.M | re.S)
    return pattern.sub('', read('README.rst'))