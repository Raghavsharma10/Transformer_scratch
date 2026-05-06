def javascript(filename, type='text/javascript'):
    '''A simple shortcut to render a ``script`` tag to a static javascript file'''
    if '?' in filename and len(filename.split('?')) is 2:
        filename, params = filename.split('?')
        return '<script type="%s" src="%s?%s"></script>' % (type, staticfiles_storage.url(filename), params)
    else:
        return '<script type="%s" src="%s"></script>' % (type, staticfiles_storage.url(filename))