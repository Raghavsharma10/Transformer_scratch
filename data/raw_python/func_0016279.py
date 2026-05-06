def _validate_model(model):
    """Check that a model configured for an index subclasses the required classes."""
    if not hasattr(model, "as_search_document"):
        raise ImproperlyConfigured("'%s' must implement `as_search_document`." % model)
    if not hasattr(model.objects, "get_search_queryset"):
        raise ImproperlyConfigured(
            "'%s.objects must implement `get_search_queryset`." % model
        )