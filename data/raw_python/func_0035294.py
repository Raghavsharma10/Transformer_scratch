def install_tools(dependencies):
    """ Install a required tool before using it, if it's missing.

        Note that C{dependencies} can be a distutils requirement,
        or a simple name from the C{tools} task configuration, or
        a (nested) list of such requirements.
    """
    tools = getattr(easy.options, "tools", {})
    for dependency in iterutil.flatten(dependencies):
        dependency = tools.get(dependency, dependency)
        try:
            pkg_resources.require(dependency)
        except pkg_resources.DistributionNotFound:
            vsh("pip", "install", "-q", dependency)
            dependency = pkg_resources.require(dependency)
            easy.info("Installed required tool %s" % (dependency,))