def rename(name):
    # type: (str) -> None
    """ Give the currently developed hotfix a new name. """
    from peltak.extra.gitflow import logic

    if name is None:
        name = click.prompt('Hotfix name')

    logic.hotfix.rename(name)