def _convert_external(bundle, name, external):
    """ Converts external documentation to resource dict ready to save to CKAN. """
    # http://docs.ckan.org/en/latest/api/#ckan.logic.action.create.resource_create
    ret = {
        'package_id': bundle.dataset.vid.lower(),
        'url': external.url,
        'description': external.description,
        'name': name,
    }

    return ret