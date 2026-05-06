def import_class(class_path):
    '''
        Imports the class for the given class name.
    '''
    module_name, class_name = class_path.rsplit(".", 1)
    module = import_module(module_name)
    claz = getattr(module, class_name)
    return claz