def get_or_create_project_venv():
    """
    Create a project-level virtualenv (if it does not already exist).

    :return: ``VirtualEnv`` object
    """
    venv_dir = get_project_venv_dir()

    if os.path.exists(venv_dir):
        return VirtualEnv(venv_dir)
    else:
        return create_project_venv()