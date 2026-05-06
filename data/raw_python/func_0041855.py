def chglog(amend: bool = False, stage: bool = False, next_version: str = None, auto_next_version: bool = False):
    """
    Writes the changelog

    Args:
        amend: amend last commit with changes
        stage: stage changes
        next_version: indicates next version
        auto_next_version: infer next version from VCS
    """
    changed_files = CTX.repo.changed_files()
    changelog_file_path: Path = config.CHANGELOG_FILE_PATH()
    changelog_file_name = changelog_file_path.name
    if changelog_file_name in changed_files:
        LOGGER.error('changelog has changed; cannot update it')
        exit(-1)
    _chglog(amend, stage, next_version, auto_next_version)