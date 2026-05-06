def generate_models(config, raml_resources):
    """ Generate model for each resource in :raml_resources:

    The DB model name is generated using singular titled version of current
    resource's url. E.g. for resource under url '/stories', model with
    name 'Story' will be generated.

    :param config: Pyramid Configurator instance.
    :param raml_resources: List of ramlfications.raml.ResourceNode.
    """
    from .models import handle_model_generation
    if not raml_resources:
        return
    for raml_resource in raml_resources:
        # No need to generate models for dynamic resource
        if is_dynamic_uri(raml_resource.path):
            continue

        # Since POST resource must define schema use only POST
        # resources to generate models
        if raml_resource.method.upper() != 'POST':
            continue

        # Generate DB model
        # If this is an attribute resource we don't need to generate model
        resource_uri = get_resource_uri(raml_resource)
        route_name = get_route_name(resource_uri)
        if not attr_subresource(raml_resource, route_name):
            log.info('Configuring model for route `{}`'.format(route_name))
            model_cls, is_auth_model = handle_model_generation(
                config, raml_resource)
            if is_auth_model:
                config.registry.auth_model = model_cls