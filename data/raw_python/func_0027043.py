def get_full_name(src):
    """Gets full class or function name."""

    if hasattr(src, "_full_name_"):
        return src._full_name_
    if hasattr(src, "is_decorator"):
        # Our own decorator or binder
        if hasattr(src, "decorator"):
            # Our own binder
            _full_name_ = str(src.decorator)
            # It's a short-living object, so we don't cache result
        else:
            # Our own decorator
            _full_name_ = str(src)
            try:
                src._full_name_ = _full_name_
            except AttributeError:
                pass
            except TypeError:
                pass
    elif hasattr(src, "im_class"):
        # Bound method
        cls = src.im_class
        _full_name_ = get_full_name(cls) + "." + src.__name__
        # It's a short-living object, so we don't cache result
    elif hasattr(src, "__module__") and hasattr(src, "__name__"):
        # Func or class
        _full_name_ = (
            ("<unknown module>" if src.__module__ is None else src.__module__)
            + "."
            + src.__name__
        )
        try:
            src._full_name_ = _full_name_
        except AttributeError:
            pass
        except TypeError:
            pass
    else:
        # Something else
        _full_name_ = str(get_original_fn(src))
    return _full_name_