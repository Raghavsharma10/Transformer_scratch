def _raiseValidationException(standardExcMsg, customExcMsg=None):
    """Raise ValidationException with standardExcMsg, unless customExcMsg is specified."""
    if customExcMsg is None:
        raise ValidationException(str(standardExcMsg))
    else:
        raise ValidationException(str(customExcMsg))