def create_model(name, *attributes, **params):
    '''Create a :class:`Model` class for objects requiring
and interface similar to :class:`StdModel`. We refers to this type
of models as :ref:`local models <local-models>` since instances of such
models are not persistent on a :class:`stdnet.BackendDataServer`.

:param name: Name of the model class.
:param attributes: positiona attribute names. These are the only attribute
    available to the model during the default constructor.
:param params: key-valued parameter to pass to the :class:`ModelMeta`
    constructor.
:return: a local :class:`Model` class.
    '''
    params['register'] = False
    params['attributes'] = attributes
    kwargs = {'manager_class': params.pop('manager_class', Manager),
              'Meta': params}
    return ModelType(name, (StdModel,), kwargs)