def _clean_lazymodule(module):
    """Removes all lazy behavior from a module's class, for loading.

    Also removes all module attributes listed under the module's class deletion
    dictionaries. Deletion dictionaries are class attributes with names
    specified in `_DELETION_DICT`.

    Parameters
    ----------
    module: LazyModule 

    Returns
    -------
    dict
        A dictionary of deleted class attributes, that can be used to reset the
        lazy state using :func:`_reset_lazymodule`.
    """
    modclass = type(module)
    _clean_lazy_submod_refs(module)

    modclass.__getattribute__ = ModuleType.__getattribute__
    modclass.__setattr__ = ModuleType.__setattr__
    cls_attrs = {}
    for cls_attr in _CLS_ATTRS:
        try:
            cls_attrs[cls_attr] = getattr(modclass, cls_attr)
            delattr(modclass, cls_attr)
        except AttributeError:
            pass
    return cls_attrs