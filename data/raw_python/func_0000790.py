def _improve_class_docs(app, cls, lines):
    """Improve the documentation of a class."""
    if issubclass(cls, models.Model):
        _add_model_fields_as_params(app, cls, lines)
    elif issubclass(cls, forms.Form):
        _add_form_fields(cls, lines)