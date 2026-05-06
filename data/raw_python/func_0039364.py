def pipupdate():
    """
    Update all currently installed pip packages
    """

    packages = [d for d in pkg_resources.working_set]
    subprocess.call('pip install --upgrade ' + ' '.join(packages))