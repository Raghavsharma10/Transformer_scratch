def git_checkout(branch_name, create=False):
    # type: (str, bool) -> None
    """ Checkout or create a given branch

    Args:
        branch_name (str):
            The name of the branch to checkout or create.
        create (bool):
            If set to **True** it will create the branch instead of checking it
            out.
    """
    log.info("Checking out <33>{}".format(branch_name))
    shell.run('git checkout {} {}'.format('-b' if create else '', branch_name))