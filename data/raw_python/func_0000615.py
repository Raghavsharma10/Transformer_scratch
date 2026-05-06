def django_js(context, jquery=True, i18n=True, csrf=True, init=True):
    '''Include Django.js javascript library in the page'''
    return {
        'js': {
            'minified': not settings.DEBUG,
            'jquery': _boolean(jquery),
            'i18n': _boolean(i18n),
            'csrf': _boolean(csrf),
            'init': _boolean(init),
        }
    }