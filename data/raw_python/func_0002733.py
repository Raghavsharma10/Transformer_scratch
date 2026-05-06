def input_relative_images(input_path, image_destination, rootname, config):
    """
    The method used to handle Input-Relative image inclusion.
    """
    log.debug('Looking for input relative images')
    input_dirname = os.path.dirname(input_path)
    for path in config.input_relative_images:
        if '*' in path:
            path = path.replace('*', rootname)
            log.debug('Wildcard expansion for image directory: {0}'.format(path))
        images = os.path.normpath(os.path.join(input_dirname, path))
        if os.path.isdir(images):
            log.info('Input-Relative image directory found: {0}'.format(images))
            shutil.copytree(images, image_destination)
            return True
    return False