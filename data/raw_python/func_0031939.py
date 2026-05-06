def createBlendedFolders():
    """Creates the standard folders for a Blended website"""
    # Create the templates folder
    create_folder(os.path.join(cwd, "templates"))

    # Create the templates/assets folder
    create_folder(os.path.join(cwd, "templates", "assets"))

    # Create the templates/assets/css folder
    create_folder(os.path.join(cwd, "templates", "assets", "css"))

    # Create the templates/assets/js folder
    create_folder(os.path.join(cwd, "templates", "assets", "js"))

    # Create the templates/assets/img folder
    create_folder(os.path.join(cwd, "templates", "assets", "img"))

    # Create the content folder
    create_folder(os.path.join(cwd, "content"))