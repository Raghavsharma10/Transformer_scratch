def django_js_init(context, jquery=False, i18n=True, csrf=True, init=True):
    '''Include Django.js javascript library initialization in the page'''
    return {
        'js': {
            'jquery': _boolean(jquery),
            'i18n': _boolean(i18n),
            'csrf': _boolean(csrf),
            'init': _boolean(init),
        }
    }