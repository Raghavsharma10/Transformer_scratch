def start(name):
    # type: (str) -> None
    """ Start working on a new feature by branching off develop.

    This will create a new branch off develop called feature/<name>.

    Args:
        name (str):
            The name of the new feature.
    """
    feature_name = 'feature/' + common.to_branch_name(name)
    develop = conf.get('git.devel_branch', 'develop')

    common.assert_on_branch(develop)
    common.git_checkout(feature_name, create=True)