def create_package_file(app, env, root_package, sub_package, private,
                        dest, suffix, dryrun, force):
    """Build the text of the file and write the file.

    :param app: the sphinx app
    :type app: :class:`sphinx.application.Sphinx`
    :param env: the jinja environment for the templates
    :type env: :class:`jinja2.Environment`
    :param root_package: the parent package
    :type root_package: :class:`str`
    :param sub_package: the package name without root
    :type sub_package: :class:`str`
    :param private: Include \"_private\" modules
    :type private: :class:`bool`
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
    logger.debug('Create package file: rootpackage %s, sub_package %s', root_package, sub_package)
    template_file = PACKAGE_TEMPLATE_NAME
    template = env.get_template(template_file)
    fn = makename(root_package, sub_package)
    var = get_context(app, root_package, sub_package, fn)
    var['ispkg'] = True
    for submod in var['submods']:
        if shall_skip(app, submod, private):
            continue
        create_module_file(app, env, fn, submod, dest, suffix, dryrun, force)
    rendered = template.render(var)
    write_file(app, fn, rendered, dest, suffix, dryrun, force)