def get_python_path(venv_path):
    """
    Get given virtual environment's `python` program path.

    :param venv_path: Virtual environment directory path.

    :return: `python` program path.
    """
    # Get `bin` directory path
    bin_path = get_bin_path(venv_path)

    # Get `python` program path
    program_path = os.path.join(bin_path, 'python')

    # If the platform is Windows
    if sys.platform.startswith('win'):
        # Add `.exe` suffix to the `python` program path
        program_path = program_path + '.exe'

    # Return the `python` program path
    return program_path