def cache_location():
    '''Cross-platform placement of cached files'''
    plat = platform.platform()
    log.debug('Platform read as: {0}'.format(plat))
    if plat.startswith('Windows'):
        log.debug('Windows platform detected')
        return os.path.join(os.environ['APPDATA'], 'OpenAccess_EPUB')
    elif plat.startswith('Darwin'):
        log.debug('Mac platform detected')
    elif plat.startswith('Linux'):
        log.debug('Linux platform detected')
    else:
        log.warning('Unhandled platform for cache_location')

    #This code is written for Linux and Mac, don't expect success for others
    path = os.path.expanduser('~')
    if path == '~':
        path = os.path.expanduser('~user')
        if path == '~user':
            log.critical('Could not resolve the correct cache location')
            sys.exit('Could not resolve the correct cache location')
    cache_loc = os.path.join(path, '.OpenAccess_EPUB')
    log.debug('Cache located: {0}'.format(cache_loc))
    return cache_loc