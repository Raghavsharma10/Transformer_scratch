def is_valid_package_module_name(name):
    """
    Test whether it's a valid package or module name.

    - a-z, 0-9, and underline
    - starts with underline or alpha letter

    valid:

    - ``a``
    - ``a.b.c``
    - ``_a``
    - ``_a._b._c``

    invalid:

    - ``A``
    - ``0``
    - ``.a``
    - ``a#b``
    """
    if "." in name:
        for part in name.split("."):
            if not is_valid_package_module_name(part):
                return False
    elif len(name):
        if name[0] not in _first_letter_for_valid_name:
            return False

        if len(set(name).difference(_char_set_for_valid_name)):
            return False
    else:
        return False
    return True