def get_install_requires():
    """Add conditional dependencies (when creating source distributions)."""
    install_requires = get_requirements('requirements.txt')
    if 'bdist_wheel' not in sys.argv:
        if sys.version_info[0] == 2:
            # On Python 2.6 and 2.7 we pull in Bazaar.
            install_requires.append('bzr >= 2.6.0')
        if sys.version_info[2:] == (2, 6):
            # On Python 2.6 we have to stick to versions of Mercurial below 4.3
            # because 4.3 drops support for Python 2.6, see the change log:
            # https://www.mercurial-scm.org/wiki/WhatsNew
            install_requires.append('mercurial >= 2.9, < 4.3')
        elif (2, 6) < sys.version_info[:2] < (3, 0):
            # On Python 2.7 we pull in Mercurial.
            install_requires.append('mercurial >= 2.9')
    return sorted(install_requires)