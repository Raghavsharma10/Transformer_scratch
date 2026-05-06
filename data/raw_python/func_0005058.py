def start(component, exact):
    # type: (str, str) -> None
    """ Create a new release branch.

    Args:
        component (str):
            Version component to bump when creating the release. Can be *major*,
            *minor* or *patch*.
        exact (str):
            The exact version to set for the release. Overrides the component
            argument. This allows to re-release a version if something went
            wrong with the release upload.
    """
    version_file = conf.get_path('version_file', 'VERSION')

    develop = conf.get('git.devel_branch', 'develop')
    common.assert_on_branch(develop)

    with conf.within_proj_dir():
        out = shell.run('git status --porcelain', capture=True).stdout
        lines = out.split(os.linesep)
        has_changes = any(
            not l.startswith('??') for l in lines if l.strip()
        )

    if has_changes:
        log.info("Cannot release: there are uncommitted changes")
        exit(1)

    old_ver, new_ver = versioning.bump(component, exact)

    log.info("Bumping package version")
    log.info("  old version: <35>{}".format(old_ver))
    log.info("  new version: <35>{}".format(new_ver))

    with conf.within_proj_dir():
        branch = 'release/' + new_ver

        common.git_checkout(branch, create=True)

        log.info("Creating commit for the release")
        shell.run('git add {ver_file} && git commit -m "{msg}"'.format(
            ver_file=version_file,
            msg="Releasing v{}".format(new_ver)
        ))