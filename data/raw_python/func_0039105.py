def prepare(tool, branch, user, included):
    """
    Prepare git for pushing
    Check that there are no permission errors
    Add necessities to git config
    Stage files
    Stage files via lfs if necessary
    Check that atleast one file is staged
    """
    with ProgressBar(_("Preparing")) as progress_bar, working_area(included) as area:
        Git.working_area = f"-C {area}"
        git = Git(Git.working_area)
        # Clone just .git folder
        try:
            _run(git.set(Git.cache)(f"clone --bare {user.repo} .git"))
        except Error:
            raise Error(_("Looks like {} isn't enabled for your account yet. "
                          "Go to https://cs50.me/authorize and make sure you accept any pending invitations!".format(tool)))

        _run(git("config --bool core.bare false"))
        _run(git(f"config --path core.worktree {area}"))

        try:
            _run(git("checkout --force {} .gitattributes".format(branch)))
        except Error:
            pass

        # Set user name/email in repo config
        _run(git(f"config user.email {shlex.quote(user.email)}"))
        _run(git(f"config user.name {shlex.quote(user.name)}"))

        # Switch to branch without checkout
        _run(git(f"symbolic-ref HEAD refs/heads/{branch}"))

        # Git add all included files
        for f in included:
            _run(git(f"add {f}"))

        # Remove gitattributes from included
        if Path(".gitattributes").exists() and ".gitattributes" in included:
            included.remove(".gitattributes")

        # Add any oversized files through git-lfs
        _lfs_add(included, git)

        progress_bar.stop()
        yield