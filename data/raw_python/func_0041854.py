def _chglog(amend: bool = False, stage: bool = False, next_version: str = None, auto_next_version: bool = False):
    """
    Writes the changelog

    Args:
        amend: amend last commit with changes
        stage: stage changes
    """
    if config.CHANGELOG_DISABLE():
        LOGGER.info('skipping changelog update as per config')
    else:
        epab.utils.ensure_exe('git')
        epab.utils.ensure_exe('gitchangelog')
        LOGGER.info('writing changelog')
        if auto_next_version:
            next_version = epab.utils.get_next_version()
        with gitchangelog_config():
            with temporary_tag(next_version):
                changelog, _ = elib_run.run('gitchangelog', mute=True)
        # changelog = changelog.encode('utf8').replace(b'\r\n', b'\n').decode('utf8')
        changelog = re.sub(BOGUS_LINE_PATTERN, '\\1\n', changelog)
        Path(config.CHANGELOG_FILE_PATH()).write_text(changelog, encoding='utf8')
        if amend:
            CTX.repo.amend_commit(
                append_to_msg='update changelog [auto]', files_to_add=str(config.CHANGELOG_FILE_PATH())
            )
        elif stage:
            CTX.repo.stage_subset(str(config.CHANGELOG_FILE_PATH()))