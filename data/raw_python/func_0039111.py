def _lfs_add(files, git):
    """
    Add any oversized files with lfs.
    Throws error if a file is bigger than 2GB or git-lfs is not installed.
    """
    # Check for large files > 100 MB (and huge files > 2 GB)
    # https://help.github.com/articles/conditions-for-large-files/
    # https://help.github.com/articles/about-git-large-file-storage/
    larges, huges = [], []
    for file in files:
        size = os.path.getsize(file)
        if size > (100 * 1024 * 1024):
            larges.append(file)
        elif size > (2 * 1024 * 1024 * 1024):
            huges.append(file)

    # Raise Error if a file is >2GB
    if huges:
        raise Error(_("These files are too large to be submitted:\n{}\n"
                      "Remove these files from your directory "
                      "and then re-run {}!").format("\n".join(huges), org))

    # Add large files (>100MB) with git-lfs
    if larges:
        # Raise Error if git-lfs not installed
        if not shutil.which("git-lfs"):
            raise Error(_("These files are too large to be submitted:\n{}\n"
                          "Install git-lfs (or remove these files from your directory) "
                          "and then re-run!").format("\n".join(larges)))

        # Install git-lfs for this repo
        _run(git("lfs install --local"))

        # For pre-push hook
        _run(git("config credential.helper cache"))

        # Rm previously added file, have lfs track file, add file again
        for large in larges:
            _run(git("rm --cached {}".format(shlex.quote(large))))
            _run(git("lfs track {}".format(shlex.quote(large))))
            _run(git("add {}".format(shlex.quote(large))))
        _run(git("add --force .gitattributes"))