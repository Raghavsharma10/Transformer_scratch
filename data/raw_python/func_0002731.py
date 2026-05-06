def move_images_to_cache(source, destination):
    """
    Handles the movement of images to the cache. Must be helpful if it finds
    that the folder for this article already exists.
    """
    if os.path.isdir(destination):
        log.debug('Cached images for this article already exist')
        return
    else:
        log.debug('Cache location: {0}'.format(destination))
        try:
            shutil.copytree(source, destination)
        except:
            log.exception('Images could not be moved to cache')
        else:
            log.info('Moved images to cache'.format(destination))