def build_spec(base_version, http_client=None, **kwargs):
    """
    Generates the Spec used to initialize a SwaggerClient, supporting mixed resource versions
    :param http_client: :class:`bravado.requests_client.RequestsClient`
    :param base_version: Version to base the spec on. Any resource without an explicit version will be this.
    :param kwargs: Explicit resource versions, by name (eg Character='v4')
    :return: :class:`bravado_core.spec.Spec`
    """
    base_spec = get_spec(base_version, http_client=http_client, config=SPEC_CONFIG)
    if kwargs:
        for resource, resource_version in kwargs.items():
            versioned_spec = get_spec(resource_version, http_client=http_client, config=SPEC_CONFIG)
            try:
                spec_resource = versioned_spec.resources[resource.capitalize()]
            except KeyError:
                raise AttributeError(
                    'Resource {0} not found on API revision {1}'.format(resource, resource_version))
            base_spec.resources[resource.capitalize()] = spec_resource
    return base_spec