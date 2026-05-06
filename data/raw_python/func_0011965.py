def create_module_file(app, env, package, module, dest, suffix, dryrun, force):
    """Build the text of the file and write the file.

    :param app: the sphinx app
    :type app: :class:`sphinx.application.Sphinx`
    :param env: the jinja environment for the templates
    :type env: :class:`jinja2.Environment`
    :param package: the package name
    :type package: :class:`str`
    :param module: the module name
    :type module: :class:`str`
    :param dest: the output directory
    :type dest: :class:`str`
    :param suffix: the file extension
    :type suffix: :class:`str`
    :param dryrun: If True, do not create any files, just log the potential location.
    :type dryrun: :class:`bool`
    :param force: Overwrite existing files
    :type force: :class:`bool`
    :returns: None
    :raises: None
    """
    logger.debug('Create module file: package %s, module %s', package, module)
    template_file = MODULE_TEMPLATE_NAME
    template = env.get_template(template_file)
    fn = makename(package, module)
    var = get_context(app, package, module, fn)
    var['ispkg'] = False
    rendered = template.render(var)
    write_file(app, makename(package, module), rendered, dest, suffix, dryrun, force)