def register_applications(self, applications, models=None, backends=None):
        '''A higher level registration functions for group of models located
on application modules.
It uses the :func:`model_iterator` function to iterate
through all :class:`Model` models available in ``applications``
and register them using the :func:`register` low level method.

:parameter applications: A String or a list of strings representing
    python dotted paths where models are implemented.
:parameter models: Optional list of models to include. If not provided
    all models found in *applications* will be included.
:parameter backends: optional dictionary which map a model or an
    application to a backend :ref:`connection string <connection-string>`.
:rtype: A list of registered :class:`Model`.

For example::


    mapper.register_application_models('mylib.myapp')
    mapper.register_application_models(['mylib.myapp', 'another.path'])
    mapper.register_application_models(pythonmodule)
    mapper.register_application_models(['mylib.myapp',pythonmodule])

'''
        return list(self._register_applications(applications, models,
                                                backends))