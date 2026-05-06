def mkdir_and_cd(dirname):
    """Change directory and/or create it if necessary."""
    if not os.path.exists(dirname):
        os.makedirs(dirname)
        os.chdir(dirname)
    else:
        os.chdir(dirname)