def start(name):
    # type: (str) -> None
    """ Start working on a new hotfix.

    This will create a new branch off master called hotfix/<name>.

    Args:
        name (str):
            The name of the new feature.
    """
    hotfix_branch = 'hotfix/' + common.to_branch_name(name)
    master = conf.get('git.master_branch', 'master')

    common.assert_on_branch(master)
    common.git_checkout(hotfix_branch, create=True)