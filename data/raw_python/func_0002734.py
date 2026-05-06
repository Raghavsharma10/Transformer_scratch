def image_cache(article_cache, img_dir):
    """
    The method to be used by get_images() for copying images out of the cache.
    """
    log.debug('Looking for image directory in the cache')
    if os.path.isdir(article_cache):
        log.info('Cached image directory found: {0}'.format(article_cache))
        shutil.copytree(article_cache, img_dir)
        return True
    return False