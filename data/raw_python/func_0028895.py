def cleanup_bundle():
    """Deletes files used for creating bundle.
        * vendored/*
        * bundle.zip
    """
    paths = ['./vendored', './bundle.zip']
    for path in paths:
        if os.path.exists(path):
            log.debug("Deleting %s..." % path)
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)