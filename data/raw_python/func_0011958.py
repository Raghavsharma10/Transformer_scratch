def write_file(app, name, text, dest, suffix, dryrun, force):
    """Write the output file for module/package <name>.

    :param app: the sphinx app
    :type app: :class:`sphinx.application.Sphinx`
    :param name: the file name without file extension
    :type name: :class:`str`
    :param text: the content of the file
    :type text: :class:`str`
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
    fname = os.path.join(dest, '%s.%s' % (name, suffix))
    if dryrun:
        logger.info('Would create file %s.' % fname)
        return
    if not force and os.path.isfile(fname):
        logger.info('File %s already exists, skipping.' % fname)
    else:
        logger.info('Creating file %s.' % fname)
        f = open(fname, 'w')
        try:
            f.write(text)
            relpath = os.path.relpath(fname, start=app.env.srcdir)
            abspath = os.sep + relpath
            docpath = app.env.relfn2path(abspath)[0]
            docpath = docpath.rsplit(os.path.extsep, 1)[0]
            logger.debug('Adding document %s' % docpath)
            app.env.found_docs.add(docpath)
        finally:
            f.close()