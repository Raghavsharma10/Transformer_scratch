def explicit_images(images, image_destination, rootname, config):
    """
    The method used to handle an explicitly defined image directory by the
    user as a parsed argument.
    """
    log.info('Explicit image directory specified: {0}'.format(images))
    if '*' in images:
        images = images.replace('*', rootname)
        log.debug('Wildcard expansion for image directory: {0}'.format(images))
    try:
        shutil.copytree(images, image_destination)
    except:
        #The following is basically a recipe for log.exception() but with a
        #CRITICAL level if the execution should be killed immediately
        #log.critical('Unable to copy from indicated directory', exc_info=True)
        log.exception('Unable to copy from indicated directory')
        return False
    else:
        return True