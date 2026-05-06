def get_model_class( klass, api = None, use_request_api = True):
    """
    Generates the Model Class based on the klass 
    loads automatically the corresponding json schema file form schemes folder
    :param klass: json schema filename
    :param use_request_api: if True autoinitializes request class if api is None
    :param api: the transportation api
                if none the default settings are taken an instantiated
    """
    if api is None and use_request_api:
        api = APIClient()
    _type = klass
    if isinstance(klass, dict):
        _type = klass['type']
    schema = loaders.load_schema_raw(_type)
    model_cls = model_factory(schema, base_class = RemoteResource)
    model_cls.__api__ = api
    return model_cls