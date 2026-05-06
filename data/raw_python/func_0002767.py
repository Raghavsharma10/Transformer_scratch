def epubcheck(epubname, config=None):
    """
    This method takes the name of an epub file as an argument. This name is
    the input for the java execution of a locally installed epubcheck-.jar. The
    location of this .jar file is configured in config.py.
    """
    if config is None:
        config = load_config_module()
    r, e = os.path.splitext(epubname)
    if not e:
        log.warning('Missing file extension, appending ".epub"')
        e = '.epub'
        epubname = r + e
    elif not e == '.epub':
        log.warning('File does not have ".epub" extension, appending it')
        epubname += '.epub'
    subprocess.call(['java', '-jar', config.epubcheck_jarfile, epubname])