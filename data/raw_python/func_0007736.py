def populate_metadata(model, MetadataClass):
    """ For a given model and metadata class, ensure there is metadata for every instance. 
    """
    content_type = ContentType.objects.get_for_model(model)
    for instance in model.objects.all():
        create_metadata_instance(MetadataClass, instance)