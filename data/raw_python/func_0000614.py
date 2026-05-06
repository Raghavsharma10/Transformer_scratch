def jquery_js(version=None, migrate=False):
    '''A shortcut to render a ``script`` tag for the packaged jQuery'''
    version = version or settings.JQUERY_VERSION
    suffix = '.min' if not settings.DEBUG else ''
    libs = [js_lib('jquery-%s%s.js' % (version, suffix))]
    if _boolean(migrate):
        libs.append(js_lib('jquery-migrate-%s%s.js' % (JQUERY_MIGRATE_VERSION, suffix)))
    return '\n'.join(libs)