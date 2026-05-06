def template_exists(form, field):
    """Form validation: check that selected template exists."""
    try:
        current_app.jinja_env.get_template(field.data)
    except TemplateNotFound:
        raise ValidationError(_("Template selected does not exist"))