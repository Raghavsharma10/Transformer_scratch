def is_exported(bundle):
    """ Returns True if dataset is already exported to CKAN. Otherwise returns False. """
    if not ckan:
        raise EnvironmentError(MISSING_CREDENTIALS_MSG)
    params = {'q': 'name:{}'.format(bundle.dataset.vid.lower())}
    resp = ckan.action.package_search(**params)
    return len(resp['results']) > 0