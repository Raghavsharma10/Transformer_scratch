def install(packagename, save, save_dev, save_test, filename):
    """
    Install the package via pip, pin the package only to requirements file.
    Use option to decide which file the package will be pinned to.
    """
    print('Installing ', packagename)
    print(sh_pip.install(packagename))
    if not filename:
        filename = get_filename(save, save_dev, save_test)
    try:
        add_requirements(packagename, filename)
    except AssertionError:
        print('Package already pinned in ', filename)