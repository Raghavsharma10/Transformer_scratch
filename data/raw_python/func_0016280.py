def _connect_signals():
    """Connect up post_save, post_delete signals for models."""
    for index in settings.get_index_names():
        for model in settings.get_index_models(index):
            _connect_model_signals(model)