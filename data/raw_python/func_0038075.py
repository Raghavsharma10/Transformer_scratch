def minimize_spec(spec_dict, operations=None, resources=None):
    """
    Trims down a source spec dict to only the operations or resources indicated.
    :param spec_dict: The source spec dict to minimize.
    :type spec_dict: dict
    :param operations: A list of opertion IDs to retain.
    :type operations: list of str
    :param resources: A list of resource names to retain.
    :type resources: list of str
    :return: Minimized swagger spec dict
    :rtype: dict
    """
    operations = operations or []
    resources = resources or []

    # keep the ugly overhead for now but only add paths we need
    minimized = {key: value for key, value in spec_dict.items() if key != 'paths'}
    minimized['paths'] = {}

    for path_name, path in spec_dict['paths'].items():
        for method, data in path.items():
            if data['operationId'] in operations or any(tag in resources for tag in data['tags']):
                if path_name not in minimized['paths']:
                    minimized['paths'][path_name] = {}
                minimized['paths'][path_name][method] = data

    return minimized