def choose_branch(exclude=None):
    # type: (List[str]) -> str
    """ Show the user a menu to pick a branch from the existing ones.

    Args:
        exclude (list[str]):
            List of branch names to exclude from the menu. By default it will
            exclude master and develop branches. To show all branches pass an
            empty array here.

    Returns:
        str: The name of the branch chosen by the user. If the user inputs an
        invalid choice, he will be asked again (and again) until he picks a
        a valid branch.
    """
    if exclude is None:
        master = conf.get('git.master_branch', 'master')
        develop = conf.get('git.devel_branch', 'develop')
        exclude = {master, develop}

    branches = list(set(git.branches()) - exclude)

    # Print the menu
    for i, branch_name in enumerate(branches):
        shell.cprint('<90>[{}] <33>{}'.format(i + 1, branch_name))

    # Get a valid choice from the user
    choice = 0
    while choice < 1 or choice > len(branches):
        prompt = "Pick a base branch from the above [1-{}]".format(
            len(branches)
        )
        choice = click.prompt(prompt, value_proc=int)
        if not (1 <= choice <= len(branches)):
            fmt = "Invalid choice {}, you must pick a number between {} and {}"
            log.err(fmt.format(choice, 1, len(branches)))

    return branches[choice - 1]