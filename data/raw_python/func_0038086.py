def newapp(path):
    """
    Generates all files for a new vodka app at the specified location.

    Will generate to current directory if no path is specified
    """

    app_path = os.path.join(VODKA_INSTALL_DIR, "resources", "blank_app")
    if not os.path.exists(path):
        os.makedirs(path)
    elif os.path.exists(os.path.join(path, "application.py")):
        click.error("There already exists a vodka app at %s, please specify a different path" % path)
    os.makedirs(os.path.join(path, "plugins"))
    shutil.copy(os.path.join(app_path, "application.py"), os.path.join(path, "application.py"))
    shutil.copy(os.path.join(app_path, "__init__.py"), os.path.join(path, "__init__.py"))
    shutil.copy(os.path.join(app_path, "plugins", "example.py"), os.path.join(path, "plugins", "example.py"))
    shutil.copy(os.path.join(app_path, "plugins", "__init__.py"), os.path.join(path, "plugins", "__init__.py"))