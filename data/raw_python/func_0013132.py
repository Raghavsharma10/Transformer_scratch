def validate_IRkernel(venv_dir):
    """Validates that this env contains an IRkernel kernel and returns info to start it


    Returns: tuple
        (ARGV, language, resource_dir)
    """
    r_exe_name = find_exe(venv_dir, "R")
    if r_exe_name is None:
        return [], None, None

    # check if this is really an IRkernel **kernel**
    import subprocess
    ressources_dir = None
    try:
        print_resources = 'cat(as.character(system.file("kernelspec", package = "IRkernel")))'
        resources_dir_bytes = subprocess.check_output([r_exe_name, '--slave', '-e', print_resources])
        resources_dir = resources_dir_bytes.decode(errors='ignore')
    except:
        # not installed? -> not useable in any case...
        return [], None, None
    argv = [r_exe_name, "--slave", "-e", "IRkernel::main()", "--args", "{connection_file}"]
    if not os.path.exists(resources_dir.strip()):
        # Fallback to our own log, but don't get the nice js goodies...
        resources_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logos", "r")
    return argv, "r", resources_dir