def codenerix(request):
    '''
    Codenerix CONTEXT
    '''
    # Get values
    DEBUG = getattr(settings, 'DEBUG', False)
    VERSION = getattr(settings, 'VERSION', _('WARNING: No version set to this code, add VERSION contant to your configuration'))
    
    # Set environment
    return {
        'DEBUG': DEBUG,
        'VERSION': VERSION,
        'CODENERIX_VERSION': __version__,
    }