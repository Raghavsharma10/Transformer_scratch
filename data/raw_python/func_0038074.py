def esi_client_factory(token=None, datasource=None, spec_file=None, version=None, **kwargs):
    """
    Generates an ESI client.
    :param token: :class:`esi.Token` used to access authenticated endpoints.
    :param datasource: Name of the ESI datasource to access.
    :param spec_file: Absolute path to a swagger spec file to load.
    :param version: Base ESI API version. Accepted values are 'legacy', 'latest', 'dev', or 'vX' where X is a number.
    :param kwargs: Explicit resource versions to build, in the form Character='v4'. Same values accepted as version.
    :return: :class:`bravado.client.SwaggerClient`

    If a spec_file is specified, specific versioning is not available. Meaning the version and resource version kwargs
    are ignored in favour of the versions available in the spec_file.
    """

    client = requests_client.RequestsClient()
    if token or datasource:
        client.authenticator = TokenAuthenticator(token=token, datasource=datasource)

    api_version = version or app_settings.ESI_API_VERSION

    if spec_file:
        return read_spec(spec_file, http_client=client)
    else:
        spec = build_spec(api_version, http_client=client, **kwargs)
        return SwaggerClient(spec)