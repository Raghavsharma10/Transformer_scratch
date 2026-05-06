def find_project_by_short_name(short_name, pbclient, all=None):
    """Return project by short_name."""
    try:
        response = pbclient.find_project(short_name=short_name, all=all)
        check_api_error(response)
        if (len(response) == 0):
            msg = '%s not found! You can use the all=1 argument to \
                   search in all the server.'
            error = 'Project Not Found'
            raise ProjectNotFound(msg, error)
        return response[0]
    except exceptions.ConnectionError:
        raise
    except ProjectNotFound:
        raise