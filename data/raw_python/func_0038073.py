def read_spec(path, http_client=None):
    """
    Reads in a swagger spec file used to initialize a SwaggerClient
    :param path: String path to local swagger spec file.
    :param http_client: :class:`bravado.requests_client.RequestsClient`
    :return: :class:`bravado_core.spec.Spec`
    """
    with open(path, 'r') as f:
        spec_dict = json.loads(f.read())

    return SwaggerClient.from_spec(spec_dict, http_client=http_client, config=SPEC_CONFIG)