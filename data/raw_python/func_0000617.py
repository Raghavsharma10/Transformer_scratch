def as_dict(self):
        '''
        Serialize the context as a dictionnary from a given request.
        '''
        data = {}
        if settings.JS_CONTEXT_ENABLED:
            for context in RequestContext(self.request):
                for key, value in six.iteritems(context):
                    if settings.JS_CONTEXT and key not in settings.JS_CONTEXT:
                        continue
                    if settings.JS_CONTEXT_EXCLUDE and key in settings.JS_CONTEXT_EXCLUDE:
                        continue
                    handler_name = 'process_%s' % key
                    if hasattr(self, handler_name):
                        handler = getattr(self, handler_name)
                        data[key] = handler(value, data)
                    elif isinstance(value, SERIALIZABLE_TYPES):
                        data[key] = value
        if settings.JS_USER_ENABLED:
            self.handle_user(data)
        return data