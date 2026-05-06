def find_exe(env_dir, name):
    """Finds a exe with that name in the environment path"""

    if platform.system() == "Windows":
        name = name + ".exe"

    # find the binary
    exe_name = os.path.join(env_dir, name)
    if not os.path.exists(exe_name):
        exe_name = os.path.join(env_dir, "bin", name)
        if not os.path.exists(exe_name):
            exe_name = os.path.join(env_dir, "Scripts", name)
            if not os.path.exists(exe_name):
                return None
    return exe_name