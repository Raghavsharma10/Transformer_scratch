def generate_server(raml_root, config):
    """ Handle server generation process.

    :param raml_root: Instance of ramlfications.raml.RootNode.
    :param config: Pyramid Configurator instance.
    """
    log.info('Server generation started')

    if not raml_root.resources:
        return

    root_resource = config.get_root_resource()
    generated_resources = {}

    for raml_resource in raml_root.resources:
        if raml_resource.path in generated_resources:
            continue

        # Get Nefertari parent resource
        parent_resource = _get_nefertari_parent_resource(
            raml_resource, generated_resources, root_resource)

        # Get generated resource and store it
        new_resource = generate_resource(
            config, raml_resource, parent_resource)
        if new_resource is not None:
            generated_resources[raml_resource.path] = new_resource