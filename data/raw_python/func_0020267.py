def model_iterator(application, include_related=True, exclude=None):
    '''A generator of :class:`StdModel` classes found in *application*.

:parameter application: A python dotted path or an iterable over python
    dotted-paths where models are defined.

Only models defined in these paths are considered.

For example::

    from stdnet.odm import model_iterator

    APPS = ('stdnet.contrib.searchengine',
            'stdnet.contrib.timeseries')

    for model in model_iterator(APPS):
        ...

'''
    if exclude is None:
        exclude = set()
    application = native_str(application)
    if ismodule(application) or isinstance(application, str):
        if ismodule(application):
            mod, application = application, application.__name__
        else:
            try:
                mod = import_module(application)
            except ImportError:
                # the module is not there
                mod = None
        if mod:
            label = application.split('.')[-1]
            try:
                mod_models = import_module('.models', application)
            except ImportError:
                mod_models = mod
            label = getattr(mod_models, 'app_label', label)
            models = set()
            for name in dir(mod_models):
                value = getattr(mod_models, name)
                meta = getattr(value, '_meta', None)
                if isinstance(value, ModelType) and meta:
                    for model in models_from_model(
                            value, include_related=include_related,
                            exclude=exclude):
                        if (model._meta.app_label == label
                                and model not in models):
                            models.add(model)
                            yield model
    else:
        for app in application:
            for m in model_iterator(app):
                yield m