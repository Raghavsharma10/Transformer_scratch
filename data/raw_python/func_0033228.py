def init(project_name):
    """Creates a new project"""

    if not VALID_PROJECT_NAME.match(project_name):
        print("Invalid project name. It may only contain letters, numbers and underscores.", file=sys.stderr)
        return

    check_path(project_name, functools.partial(shutil.copytree, skeleton_path("plugin")))
    check_path("static", os.mkdir)
    check_path("templates", os.mkdir)
    check_path("config.py", functools.partial(config_maker, project_name))