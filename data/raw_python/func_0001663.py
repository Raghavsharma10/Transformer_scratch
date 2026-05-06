def activate(lancet, method, project):
    """Switch to this project."""
    with taskstatus("Looking up project") as ts:
        if method == "key":
            func = get_project_keys
        elif method == "dir":
            func = get_project_keys

        for key, project_path in func(lancet):
            if key.lower() == project.lower():
                break
        else:
            ts.abort(
                'Project "{}" not found (using {}-based lookup)',
                project,
                method,
            )

    # Load the configuration
    config = load_config(os.path.join(project_path, LOCAL_CONFIG))

    # cd to the project directory
    lancet.defer_to_shell("cd", project_path)

    # Activate virtualenv
    venv = config.get("lancet", "virtualenv", fallback=None)
    if venv:
        venv_path = os.path.join(project_path, os.path.expanduser(venv))
        activate_script = os.path.join(venv_path, "bin", "activate")
        lancet.defer_to_shell("source", activate_script)
    else:
        if "VIRTUAL_ENV" in os.environ:
            lancet.defer_to_shell("deactivate")