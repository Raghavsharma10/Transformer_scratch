def validate(style):
    """Check `style` against pyout.styling.schema.

    Parameters
    ----------
    style : dict
        Style object to validate.

    Raises
    ------
    StyleValidationError if `style` is not valid.
    """
    try:
        import jsonschema
    except ImportError:
        return

    try:
        jsonschema.validate(style, schema)
    except jsonschema.ValidationError as exc:
        new_exc = StyleValidationError(exc)
        # Don't dump the original jsonschema exception because it is already
        # included in the StyleValidationError's message.
        new_exc.__cause__ = None
        raise new_exc