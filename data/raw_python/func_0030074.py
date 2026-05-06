def update_repository_configuration(id, external_repository=None, prebuild_sync=None):
    """
    Update an existing RepositoryConfiguration with new information
    """
    to_update_id = id

    rc_to_update = pnc_api.repositories.get_specific(id=to_update_id).content

    if external_repository is None:
        external_repository = rc_to_update.external_url
    else:
        rc_to_update.external_url = external_repository

    if prebuild_sync is not None:
        rc_to_update.pre_build_sync_enabled = prebuild_sync

    if not external_repository and prebuild_sync:
        logging.error("You cannot enable prebuild sync without external repository")
        return

    response = utils.checked_api_call(pnc_api.repositories, 'update', id=to_update_id, body=rc_to_update)
    if response:
        return response.content