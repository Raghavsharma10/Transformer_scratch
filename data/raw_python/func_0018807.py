def autodoc_applicationmodel(module):
    """Improves the docstrings of application models when called
    at the bottom of the respective module.

    |autodoc_applicationmodel| requires, similar to
    |autodoc_basemodel|, that both the application model and its
    base model are defined in the conventional way.
    """
    autodoc_tuple2doc(module)
    name_applicationmodel = module.__name__
    name_basemodel = name_applicationmodel.split('_')[0]
    module_basemodel = importlib.import_module(name_basemodel)
    substituter = Substituter(module_basemodel.substituter)
    substituter.add_module(module)
    substituter.update_masters()
    module.substituter = substituter