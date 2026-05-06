def TemplateValidator(value):
    """Try to compile a string into a Django template"""

    try:
        Template(value)
    except Exception as e:
        raise ValidationError(
            _("Cannot compile template (%(exception)s)"),
            params={"exception": e}
        )