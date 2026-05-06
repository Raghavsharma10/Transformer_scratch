def class_from_string(name):
    '''
    Get a python class object from its name
    '''
    module_name, class_name = name.rsplit('.', 1)
    __import__(module_name)
    module = sys.modules[module_name]
    return getattr(module, class_name)