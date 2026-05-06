def versioning_model_classname(manager, model):
    """Get the name of the versioned model class."""
    if manager.options.get('use_module_name', True):
        return '%s%sVersion' % (
            model.__module__.title().replace('.', ''), model.__name__)
    else:
        return '%sVersion' % (model.__name__,)