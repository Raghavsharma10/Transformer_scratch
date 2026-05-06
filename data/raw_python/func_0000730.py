def prepare_relationship(config, model_name, raml_resource):
    """ Create referenced model if it doesn't exist.

    When preparing a relationship, we check to see if the model that will be
    referenced already exists. If not, it is created so that it will be possible
    to use it in a relationship. Thus the first usage of this model in RAML file
    must provide its schema in POST method resource body schema.

    :param model_name: Name of model which should be generated.
    :param raml_resource: Instance of ramlfications.raml.ResourceNode for
        which :model_name: will be defined.
    """
    if get_existing_model(model_name) is None:
        plural_route = '/' + pluralize(model_name.lower())
        route = '/' + model_name.lower()
        for res in raml_resource.root.resources:
            if res.method.upper() != 'POST':
                continue
            if res.path.endswith(plural_route) or res.path.endswith(route):
                break
        else:
            raise ValueError('Model `{}` used in relationship is not '
                             'defined'.format(model_name))
        setup_data_model(config, res, model_name)