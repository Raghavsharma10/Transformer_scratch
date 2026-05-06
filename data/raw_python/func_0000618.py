def process_LANGUAGE_CODE(self, language_code, data):
        '''
        Fix language code when set to non included default `en`
        and add the extra variables ``LANGUAGE_NAME`` and ``LANGUAGE_NAME_LOCAL``.
        '''
        # Dirty hack to fix non included default
        language_code = 'en-us' if language_code == 'en' else language_code
        language = translation.get_language_info('en' if language_code == 'en-us' else language_code)
        if not settings.JS_CONTEXT or 'LANGUAGE_NAME' in settings.JS_CONTEXT \
            or (settings.JS_CONTEXT_EXCLUDE and 'LANGUAGE_NAME' in settings.JS_CONTEXT_EXCLUDE):
            data['LANGUAGE_NAME'] = language['name']
        if not settings.JS_CONTEXT or 'LANGUAGE_NAME_LOCAL' in settings.JS_CONTEXT \
            or (settings.JS_CONTEXT_EXCLUDE and 'LANGUAGE_NAME_LOCAL' in settings.JS_CONTEXT_EXCLUDE):
            data['LANGUAGE_NAME_LOCAL'] = language['name_local']
        return language_code