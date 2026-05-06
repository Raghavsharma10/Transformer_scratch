def create_repository_configuration(repository, no_sync=False):
    """
    Create a new RepositoryConfiguration. If the provided repository URL is for external repository, it is cloned into internal one.
    :return BPM Task ID of the new RepositoryConfiguration creation
    """
    repo = create_repository_configuration_raw(repository, no_sync)
    if repo:
        return utils.format_json(repo)