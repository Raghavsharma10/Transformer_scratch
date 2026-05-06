def main(app):
    """Parse the config of the app and initiate the generation process

    :param app: the sphinx app
    :type app: :class:`sphinx.application.Sphinx`
    :returns: None
    :rtype: None
    :raises: None
    """
    c = app.config
    src = c.jinjaapi_srcdir

    if not src:
        return

    suffix = "rst"

    out = c.jinjaapi_outputdir or app.env.srcdir

    if c.jinjaapi_addsummarytemplate:
        tpath = pkg_resources.resource_filename(__package__, AUTOSUMMARYTEMPLATE_DIR)
        c.templates_path.append(tpath)

    tpath = pkg_resources.resource_filename(__package__, TEMPLATE_DIR)
    c.templates_path.append(tpath)

    prepare_dir(app, out, not c.jinjaapi_nodelete)
    generate(app, src, out,
             exclude=c.jinjaapi_exclude_paths,
             force=c.jinjaapi_force,
             followlinks=c.jinjaapi_followlinks,
             dryrun=c.jinjaapi_dryrun,
             private=c.jinjaapi_includeprivate,
             suffix=suffix,
             template_dirs=c.templates_path)