def build(
    c,
    clean=False,
    browse=False,
    nitpick=False,
    opts=None,
    source=None,
    target=None,
):
    """
    Build the project's Sphinx docs.
    """
    if clean:
        _clean(c)
    if opts is None:
        opts = ""
    if nitpick:
        opts += " -n -W -T"
    cmd = "sphinx-build{0} {1} {2}".format(
        (" " + opts) if opts else "",
        source or c.sphinx.source,
        target or c.sphinx.target,
    )
    c.run(cmd, pty=True)
    if browse:
        _browse(c)