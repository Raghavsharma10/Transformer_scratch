def raise_validator_errors(validator):
    """
    Raise any errors associated with the validator.

    Parameters
    ----------
    validator : :class:`cerberus.Validator`
        Validator

    Raises
    ------
    ValueError
        Raised if errors existed on `validator`.
        Message describing each error and information
        associated with the configuration option
        causing the error.
    """

    if len(validator._errors) == 0:
        return

    def _path_str(path, name=None):
        """ String of the document/schema path. `cfg["foo"]["bar"]` """
        L = [name] if name is not None else []
        L.extend('["%s"]' % p for p in path)
        return "".join(L)

    def _path_leaf(path, dicts):
        """ Dictionary Leaf of the schema/document given the path """
        for p in path:
            dicts = dicts[p]

        return dicts

    wrap = partial(textwrap.wrap, initial_indent=' '*4,
                                subsequent_indent=' '*8)

    msg = ["There were configuration errors:"]

    for e in validator._errors:
        schema_leaf = _path_leaf(e.document_path, validator.schema)
        doc_str = _path_str(e.document_path, "cfg")

        msg.append("Invalid configuration option %s == '%s'." % (doc_str, e.value))

        try:
            otype = schema_leaf["type"]
            msg.extend(wrap("Type must be '%s'." % otype))
        except KeyError:
            pass

        try:
            allowed = schema_leaf["allowed"]
            msg.extend(wrap("Allowed values are '%s'." % allowed))
        except KeyError:
            pass

        try:
            description = schema_leaf["__description__"]
            msg.extend(wrap("Description: %s" % description))
        except KeyError:
            pass

    raise ValueError("\n".join(msg))