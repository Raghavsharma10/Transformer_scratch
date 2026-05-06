def _to_swagger(base=None, description=None, resource=None, options=None):
    # type: (Dict[str, str], str, Resource, Dict[str, str]) -> Dict[str, str]
    """
    Common to swagger definition.

    :param base: The base dict.
    :param description: An optional description.
    :param resource: An optional resource.
    :param options: Any additional options

    """
    definition = dict_filter(base or {}, options or {})

    if description:
        definition['description'] = description.format(
            name=getmeta(resource).name if resource else "UNKNOWN"
        )

    if resource:
        definition['schema'] = {
            '$ref': '#/definitions/{}'.format(getmeta(resource).resource_name)
        }

    return definition