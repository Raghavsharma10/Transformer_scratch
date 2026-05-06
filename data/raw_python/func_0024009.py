def context_processors_update(context, request):
    '''
    Update context with context_processors from settings
    Usage:
        from codenerix.helpers import context_processors_update
        context_processors_update(context, self.request)
    '''
    for template in settings.TEMPLATES:
        for context_processor in template['OPTIONS']['context_processors']:
            path = context_processor.split('.')
            name = path.pop(-1)
            processor = getattr(importlib.import_module('.'.join(path)), name, None)
            if processor:
                context.update(processor(request))
    return context