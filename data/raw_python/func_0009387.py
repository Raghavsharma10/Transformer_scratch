def sites(c):
    """
    Build both doc sites w/ maxed nitpicking.
    """
    # TODO: This is super lolzy but we haven't actually tackled nontrivial
    # in-Python task calling yet, so we do this to get a copy of 'our' context,
    # which has been updated with the per-collection config data of the
    # docs/www subcollections.
    docs_c = Context(config=c.config.clone())
    www_c = Context(config=c.config.clone())
    docs_c.update(**docs.configuration())
    www_c.update(**www.configuration())
    # Must build both normally first to ensure good intersphinx inventory files
    # exist =/ circular dependencies ahoy! Do it quietly to avoid pulluting
    # output; only super-serious errors will bubble up.
    # TODO: wants a 'temporarily tweak context settings' contextmanager
    # TODO: also a fucking spinner cuz this confuses me every time I run it
    # when the docs aren't already prebuilt
    docs_c["run"].hide = True
    www_c["run"].hide = True
    docs["build"](docs_c)
    www["build"](www_c)
    docs_c["run"].hide = False
    www_c["run"].hide = False
    # Run the actual builds, with nitpick=True (nitpicks + tracebacks)
    docs["build"](docs_c, nitpick=True)
    www["build"](www_c, nitpick=True)