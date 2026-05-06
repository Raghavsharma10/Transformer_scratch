def start(name):
    # type: (str) -> None
    """ Start working on a new feature by branching off develop.

    This will create a new branch off develop called feature/<name>.

    Args:
        name (str):
            The name of the new feature.
    """
    branch = git.current_branch(refresh=True)
    task_branch = 'task/' + common.to_branch_name(name)

    if branch.type not in ('feature', 'hotfix'):
        log.err("Task branches can only branch off <33>feature<32> or "
                "<33>hotfix<32> branches")
        sys.exit(1)

    common.git_checkout(task_branch, create=True)