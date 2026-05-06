def handle_model_generation(config, raml_resource):
    """ Generates model name and runs `setup_data_model` to get
    or generate actual model class.

    :param raml_resource: Instance of ramlfications.raml.ResourceNode.
    """
    model_name = generate_model_name(raml_resource)
    try:
        return setup_data_model(config, raml_resource, model_name)
    except ValueError as ex:
        raise ValueError('{}: {}'.format(model_name, str(ex)))