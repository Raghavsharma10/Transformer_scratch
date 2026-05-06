def get_all_classes(module_name):
    """Load all non-abstract classes from package"""
    module = importlib.import_module(module_name)
    return getmembers(module, lambda m: isclass(m) and not isabstract(m))