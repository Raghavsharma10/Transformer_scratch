def lint(exclude, skip_untracked, commit_only):
    # type: (List[str], bool, bool) -> None
    """ Lint python files.

    Args:
        exclude (list[str]):
            A list of glob string patterns to test against. If the file/path
            matches any of those patters, it will be filtered out.
        skip_untracked (bool):
            If set to **True** it will skip all files not tracked by git.
        commit_only (bool):
            Only lint files that are staged for commit.
    """
    exclude = list(exclude) + conf.get('lint.exclude', [])
    runner = LintRunner(exclude, skip_untracked, commit_only)

    if not runner.run():
        exit(1)