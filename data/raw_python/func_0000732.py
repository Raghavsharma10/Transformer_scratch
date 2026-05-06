def setup_data_model(config, raml_resource, model_name):
    """ Setup storage/data model and return generated model class.

    Process follows these steps:
      * Resource schema is found and restructured by `resource_schema`.
      * Model class is generated from properties dict using util function
        `generate_model_cls`.

    :param raml_resource: Instance of ramlfications.raml.ResourceNode.
    :param model_name: String representing model name.
    """
    model_cls = get_existing_model(model_name)
    schema = resource_schema(raml_resource)

    if not schema:
        raise Exception('Missing schema for model `{}`'.format(model_name))

    if model_cls is not None:
        return model_cls, schema.get('_auth_model', False)

    log.info('Generating model class `{}`'.format(model_name))
    return generate_model_cls(
        config,
        schema=schema,
        model_name=model_name,
        raml_resource=raml_resource,
    )