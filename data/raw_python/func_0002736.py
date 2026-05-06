def make_image_cache(img_cache):
    """
    Initiates the image cache if it does not exist
    """
    log.info('Initiating the image cache at {0}'.format(img_cache))
    if not os.path.isdir(img_cache):
        utils.mkdir_p(img_cache)
        utils.mkdir_p(os.path.join(img_cache, '10.1371'))
        utils.mkdir_p(os.path.join(img_cache, '10.3389'))