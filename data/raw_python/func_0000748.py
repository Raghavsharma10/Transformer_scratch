def generate_model_name(raml_resource):
    """ Generate model name.

    :param raml_resource: Instance of ramlfications.raml.ResourceNode.
    """
    resource_uri = get_resource_uri(raml_resource).strip('/')
    resource_uri = re.sub('\W', ' ', resource_uri)
    model_name = inflection.titleize(resource_uri)
    return inflection.singularize(model_name).replace(' ', '')