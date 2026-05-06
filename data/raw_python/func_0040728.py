def reqs(amend: bool = False, stage: bool = False):
    """
    Write requirements files

    Args:
        amend: amend last commit with changes
        stage: stage changes
    """
    changed_files = CTX.repo.changed_files()
    if 'requirements.txt' in changed_files or 'requirements-dev.txt' in changed_files:
        LOGGER.error('Requirements have changed; cannot update them')
        sys.exit(-1)
    _write_reqs(amend, stage)