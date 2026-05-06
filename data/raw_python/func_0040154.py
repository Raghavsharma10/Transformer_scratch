def match_modules(allowed_modules):
    """Creates a matcher that matches a list/set/tuple of allowed modules."""
    cleaned_allowed_modules = [
        utils.mod_to_mod_name(tmp_mod)
        for tmp_mod in allowed_modules
    ]
    cleaned_split_allowed_modules = [
        tmp_mod.split(".")
        for tmp_mod in cleaned_allowed_modules
    ]
    cleaned_allowed_modules = []
    del cleaned_allowed_modules

    def matcher(cause):
        cause_cls = None
        cause_type_name = cause.exception_type_names[0]
        # Rip off the class name (usually at the end).
        cause_type_name_pieces = cause_type_name.split(".")
        cause_type_name_mod_pieces = cause_type_name_pieces[0:-1]
        # Do any modules provided match the provided causes module?
        mod_match = any(
            utils.array_prefix_matches(mod_pieces,
                                       cause_type_name_mod_pieces)
            for mod_pieces in cleaned_split_allowed_modules)
        if mod_match:
            cause_cls = importutils.import_class(cause_type_name)
            cause_cls = ensure_base_exception(cause_type_name, cause_cls)
        return cause_cls

    return matcher