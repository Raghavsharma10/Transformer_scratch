def create_project_venv():
    """
    Create a project-level virtualenv.

    :raises: if virtualenv exists already
    :return: ``VirtualEnv`` object
    """
    print('... creating project-level virtualenv')
    venv_dir = get_project_venv_dir()

    if os.path.exists(venv_dir):
        raise Exception('ERROR: virtualenv already exists!')

    use_venv_module = sys.version_info >= (3, 0) and 'APE_USE_VIRTUALENV' not in os.environ

    VirtualEnv.create_virtualenv(venv_dir, use_venv_module=use_venv_module)

    print('... virtualenv successfully created')
    return VirtualEnv(venv_dir)