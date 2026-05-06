def preset_find(query):
    """Find preset from hub

    \b
    $ be find ad
    https://github.com/mottosso/be-ad.git

    """

    if self.isactive():
        lib.echo("ERROR: Exit current project first")
        sys.exit(lib.USER_ERROR)

    found = _extern.github_presets().get(query)
    if found:
        lib.echo(found)
    else:
        lib.echo("Unable to locate preset \"%s\"" % query)