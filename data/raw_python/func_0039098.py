def push(tool, slug, config_loader, prompt=lambda included, excluded: True):
    """
    Push to github.com/org/repo=username/slug if tool exists.
    Returns username, commit hash
    """
    check_dependencies()

    org, (included, excluded) = connect(slug, config_loader)

    with authenticate(org) as user, prepare(tool, slug, user, included):
        if prompt(included, excluded):
            return upload(slug, user, tool)
        else:
            raise Error(_("No files were submitted."))