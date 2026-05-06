def hvenvup(package, directory):
    """
    Install a new version of a package in the hitch venv.
    """
    pip = Command(DIR.gen.joinpath("hvenv", "bin", "pip"))
    pip("uninstall", package, "-y").run()
    pip("install", DIR.project.joinpath(directory).abspath()).run()