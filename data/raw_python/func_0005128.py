def get_parent_obj(obj):
    """ Gets the name of the object containing @obj and returns as a string

        @obj: any python object

        -> #str parent object name or None
        ..
            from vital.debug import get_parent_obj

            get_parent_obj(get_parent_obj)
            # -> <module 'vital.debug' from>
        ..
    """
    try:
        cls = get_class_that_defined_method(obj)
        if cls and cls != obj:
            return cls
    except AttributeError:
        pass
    if hasattr(obj, '__module__') and obj.__module__:
        try:
            module = locate(obj.__module__)
            assert module is not obj
            return module
        except Exception:
            try:
                module = module.__module__.split('.')[:-1]
                if len(module):
                    return locate(module)
            except Exception:
                pass
    elif hasattr(obj, '__objclass__') and obj.__objclass__:
        return obj.__objclass__
    try:
        assert hasattr(obj, '__qualname__') or hasattr(obj, '__name__')
        objname = obj.__qualname__ if hasattr(obj, '__qualname__') \
            else obj.__name__
        objname = objname.split(".")
        assert len(objname) > 1
        return locate(".".join(objname[:-1]))
    except Exception:
        try:
            module = importlib.import_module(".".join(objname[:-1]))
            return module
        except Exception:
            pass
    return None