def remove(packagename, save, save_dev, save_test, filename):
    """
    Uninstall the package and remove it from requirements file.
    """
    print(sh_pip.uninstall(packagename, "-y"))
    if not filename:
        filename = get_filename(save, save_dev, save_test)
    remove_requirements(packagename, filename)