def get_repository_configuration(id):
    """
    Retrieve a specific RepositoryConfiguration
    """

    response = utils.checked_api_call(pnc_api.repositories, 'get_specific', id=id)
    if response:
        return response.content