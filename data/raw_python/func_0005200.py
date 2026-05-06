def unstaged():
    # type: () -> List[str]
    """ Return a list of unstaged files in the project repository.

    Returns:
        list[str]: The list of files not tracked by project git repo.
    """
    with conf.within_proj_dir():
        status = shell.run(
            'git status --porcelain',
            capture=True,
            never_pretend=True
        ).stdout
        results = []

        for file_status in status.split(os.linesep):
            if file_status.strip() and file_status[0] == ' ':
                results.append(file_status[3:].strip())

        return results