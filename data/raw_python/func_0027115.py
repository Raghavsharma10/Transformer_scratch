def copy_public_attrs(source_obj, dest_obj):
    """Shallow copies all public attributes from source_obj to dest_obj.

    Overwrites them if they already exist.

    """
    for name, value in inspect.getmembers(source_obj):
        if not any(name.startswith(x) for x in ["_", "func", "im"]):
            setattr(dest_obj, name, value)