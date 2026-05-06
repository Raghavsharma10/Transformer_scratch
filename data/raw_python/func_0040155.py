def match_classes(allowed_classes):
    """Creates a matcher that matches a list/tuple of allowed classes."""
    cleaned_allowed_classes = [
        utils.cls_to_cls_name(tmp_cls)
        for tmp_cls in allowed_classes
    ]

    def matcher(cause):
        cause_cls = None
        cause_type_name = cause.exception_type_names[0]
        try:
            cause_cls_idx = cleaned_allowed_classes.index(cause_type_name)
        except ValueError:
            pass
        else:
            cause_cls = allowed_classes[cause_cls_idx]
            if not isinstance(cause_cls, type):
                cause_cls = importutils.import_class(cause_cls)
            cause_cls = ensure_base_exception(cause_type_name, cause_cls)
        return cause_cls

    return matcher