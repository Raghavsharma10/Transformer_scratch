def validate_IPykernel(venv_dir):
    """Validates that this env contains an IPython kernel and returns info to start it


    Returns: tuple
        (ARGV, language, resource_dir)
    """
    python_exe_name = find_exe(venv_dir, "python")
    if python_exe_name is None:
        python_exe_name = find_exe(venv_dir, "python2")
    if python_exe_name is None:
        python_exe_name = find_exe(venv_dir, "python3")
    if python_exe_name is None:
        return [], None, None

    # Make some checks for ipython first, because calling the import is expensive
    if find_exe(venv_dir, "ipython") is None:
        if find_exe(venv_dir, "ipython2") is None:
            if find_exe(venv_dir, "ipython3") is None:
                return [], None, None

    # check if this is really an ipython **kernel**
    import subprocess
    try:
        subprocess.check_call([python_exe_name, '-c', '"import ipykernel"'])
    except:
        # not installed? -> not useable in any case...
        return [], None, None
    argv = [python_exe_name, "-m", "ipykernel", "-f", "{connection_file}"]
    resources_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logos", "python")
    return argv, "python", resources_dir