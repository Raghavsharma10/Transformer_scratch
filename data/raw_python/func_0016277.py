def _validate_config(strict=False):
    """Validate settings.SEARCH_SETTINGS."""
    for index in settings.get_index_names():
        _validate_mapping(index, strict=strict)
        for model in settings.get_index_models(index):
            _validate_model(model)
    if settings.get_setting("update_strategy", "full") not in ["full", "partial"]:
        raise ImproperlyConfigured(
            "Invalid SEARCH_SETTINGS: 'update_strategy' value must be 'full' or 'partial'."
        )