def resource_schema(raml_resource):
    """ Get schema properties of RAML resource :raml_resource:.

    Must be called with RAML resource that defines body schema. First
    body that defines schema is used. Schema is converted on return using
    'convert_schema'.

    :param raml_resource: Instance of ramlfications.raml.ResourceNode of
        POST method.
    """
    # NOTE: Must be called with resource that defines body schema
    log.info('Searching for model schema')
    if not raml_resource.body:
        raise ValueError('RAML resource has no body to setup database '
                         'schema from')

    for body in raml_resource.body:
        if body.schema:
            return convert_schema(body.schema, body.mime_type)
    log.debug('No model schema found.')