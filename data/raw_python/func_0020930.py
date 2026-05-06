def get_file_path(filename, local=True, relative_to_module=None, my_dir=my_dir):
    """
    Look for an existing path matching filename.
    Try to resolve relative to the module location if the path cannot by found
    using "normal" resolution.
    """
    # override my_dir if module is provided
    if relative_to_module is not None:
        my_dir = os.path.dirname(relative_to_module.__file__)
    user_path = result = filename
    if local:
        user_path = os.path.expanduser(filename)
        result = os.path.abspath(user_path)
        if os.path.exists(result):
            return result  # The file was found normally
    # otherwise look relative to the module.
    result = os.path.join(my_dir, filename)
    assert os.path.exists(result), "no such file " + repr((filename, result, user_path))
    return result