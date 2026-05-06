def _all_default(d, default, seen=None):
    """
    ANY VALUE NOT SET WILL BE SET BY THE default
    THIS IS RECURSIVE
    """
    if default is None:
        return
    if _get(default, CLASS) is Data:
        default = object.__getattribute__(default, SLOT)  # REACH IN AND GET THE dict
        # Log = _late_import()
        # Log.error("strictly dict (or object) allowed: got {{type}}", type=_get(default, CLASS).__name__)

    for k, default_value in default.items():
        default_value = unwrap(default_value)  # TWO DIFFERENT Dicts CAN SHARE id() BECAUSE THEY ARE SHORT LIVED
        existing_value = _get_attr(d, [k])

        if existing_value == None:
            if default_value != None:
                if _get(default_value, CLASS) in data_types:
                    df = seen.get(id(default_value))
                    if df is not None:
                        _set_attr(d, [k], df)
                    else:
                        copy_dict = {}
                        seen[id(default_value)] = copy_dict
                        _set_attr(d, [k], copy_dict)
                        _all_default(copy_dict, default_value, seen)
                else:
                    # ASSUME PRIMITIVE (OR LIST, WHICH WE DO NOT COPY)
                    try:
                        _set_attr(d, [k], default_value)
                    except Exception as e:
                        if PATH_NOT_FOUND not in e:
                            get_logger().error("Can not set attribute {{name}}", name=k, cause=e)
        elif is_list(existing_value) or is_list(default_value):
            _set_attr(d, [k], None)
            _set_attr(d, [k], listwrap(existing_value) + listwrap(default_value))
        elif (hasattr(existing_value, "__setattr__") or _get(existing_value, CLASS) in data_types) and _get(default_value, CLASS) in data_types:
            df = seen.get(id(default_value))
            if df is not None:
                _set_attr(d, [k], df)
            else:
                seen[id(default_value)] = existing_value
                _all_default(existing_value, default_value, seen)