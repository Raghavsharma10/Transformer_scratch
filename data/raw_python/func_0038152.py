def get_schema_model():
    """
    Returns the schema model that is active in this project.
    """
    try:
        return django_apps.get_model(settings.POSTGRES_SCHEMA_MODEL, require_ready=False)
    except ValueError:
        raise ImproperlyConfigured("POSTGRES_SCHEMA_MODEL must be of the form 'app_label.model_name'")
    except LookupError:
        raise ImproperlyConfigured(
            "POSTGRES_SCHEMA_MODEL refers to model '%s' that has not been installed" % settings.POSTGRES_SCHEMA_MODEL
        )