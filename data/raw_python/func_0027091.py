def deserialize_class(serilalized_cls):
    """ Deserialize Python class """
    module_name, cls_name = serilalized_cls.split(':')
    module = importlib.import_module(module_name)
    return getattr(module, cls_name)