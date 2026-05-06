def get_linked_metadata(obj, name=None, context=None, site=None, language=None):
    """ Gets metadata linked from the given object. """
    # XXX Check that 'modelinstance' and 'model' metadata are installed in backends
    # I believe that get_model() would return None if not
    Metadata = _get_metadata_model(name)
    InstanceMetadata = Metadata._meta.get_model('modelinstance')
    ModelMetadata = Metadata._meta.get_model('model')
    content_type = ContentType.objects.get_for_model(obj)
    instances = []
    if InstanceMetadata is not None:
        try:
            instance_md = InstanceMetadata.objects.get(_content_type=content_type, _object_id=obj.pk)
        except InstanceMetadata.DoesNotExist:
            instance_md = InstanceMetadata(_content_object=obj)
        instances.append(instance_md)
    if ModelMetadata is not None:
        try:
            model_md = ModelMetadata.objects.get(_content_type=content_type)
        except ModelMetadata.DoesNotExist:
            model_md = ModelMetadata(_content_type=content_type)
        instances.append(model_md)    
    return FormattedMetadata(Metadata, instances, '', site, language)