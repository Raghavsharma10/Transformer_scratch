def find_applications_on_system():
    """
    Collect maya version from Autodesk PATH if exists, else try looking
    for custom executable paths from config file.
    """
    # First we collect maya versions from the Autodesk folder we presume
    # is addeed to the system environment "PATH"
    path_env = os.getenv('PATH').split(os.pathsep)
    versions = {}
    for each in path_env:
        path = Path(os.path.expandvars(each))
        if not path.exists():
            continue
        if path.name.endswith(DEVELOPER_NAME):
            if not path.exists():
                continue
            versions.update(get_version_exec_mapping_from_path(path))
    return versions