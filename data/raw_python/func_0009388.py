def watch_docs(c):
    """
    Watch both doc trees & rebuild them if files change.

    This includes e.g. rebuilding the API docs if the source code changes;
    rebuilding the WWW docs if the README changes; etc.

    Reuses the configuration values ``packaging.package`` or ``tests.package``
    (the former winning over the latter if both defined) when determining which
    source directory to scan for API doc updates.
    """
    # TODO: break back down into generic single-site version, then create split
    # tasks as with docs/www above. Probably wants invoke#63.

    # NOTE: 'www'/'docs' refer to the module level sub-collections. meh.

    # Readme & WWW triggers WWW
    www_c = Context(config=c.config.clone())
    www_c.update(**www.configuration())
    www_handler = make_handler(
        ctx=www_c,
        task_=www["build"],
        regexes=[r"\./README.rst", r"\./sites/www"],
        ignore_regexes=[r".*/\..*\.swp", r"\./sites/www/_build"],
    )

    # Code and docs trigger API
    docs_c = Context(config=c.config.clone())
    docs_c.update(**docs.configuration())
    regexes = [r"\./sites/docs"]
    package = c.get("packaging", {}).get("package", None)
    if package is None:
        package = c.get("tests", {}).get("package", None)
    if package:
        regexes.append(r"\./{}/".format(package))
    api_handler = make_handler(
        ctx=docs_c,
        task_=docs["build"],
        regexes=regexes,
        ignore_regexes=[r".*/\..*\.swp", r"\./sites/docs/_build"],
    )

    observe(www_handler, api_handler)