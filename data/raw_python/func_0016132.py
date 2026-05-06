def _reset_lazymodule(module, cls_attrs):
    """Resets a module's lazy state from cached data.

    """
    modclass = type(module)
    del modclass.__getattribute__
    del modclass.__setattr__
    try:
        del modclass._LOADING
    except AttributeError:
        pass
    for cls_attr in _CLS_ATTRS:
        try:
            setattr(modclass, cls_attr, cls_attrs[cls_attr])
        except KeyError:
            pass
    _reset_lazy_submod_refs(module)