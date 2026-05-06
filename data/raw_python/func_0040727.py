def _write_reqs(amend: bool = False, stage: bool = False):
    """
    Writes the requirement files

    Args:
        amend: amend last commit with changes
        stage: stage changes
    """
    LOGGER.info('writing requirements')

    base_cmd = 'pipenv lock -r'
    _write_reqs_file(f'{base_cmd}', 'requirements.txt')
    _write_reqs_file(f'{base_cmd} -d', 'requirements-dev.txt')
    files_to_add = ['Pipfile', 'requirements.txt', 'requirements-dev.txt']

    if amend:
        CTX.repo.amend_commit(append_to_msg='update requirements [auto]', files_to_add=files_to_add)
    elif stage:
        CTX.repo.stage_subset(*files_to_add)