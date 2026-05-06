def populate_all_metadata():
    """ Create metadata instances for all models in seo_models if empty.
        Once you have created a single metadata instance, this will not run.
        This is because it is a potentially slow operation that need only be
        done once. If you want to ensure that everything is populated, run the
        populate_metadata management command.
    """
    for Metadata in registry.values():
        InstanceMetadata = Metadata._meta.get_model('modelinstance')
        if InstanceMetadata is not None:
            for model in Metadata._meta.seo_models:
                populate_metadata(model, InstanceMetadata)