def call_unrar(params):
    """Calls rar/unrar command line executable, returns stdout pipe"""
    global rar_executable_cached
    if rar_executable_cached is None:
        for command in ('unrar', 'rar'):
            try:
                subprocess.Popen([command], stdout=subprocess.PIPE)
                rar_executable_cached = command
                break
            except OSError:
                pass
        if rar_executable_cached is None:
            raise UnpackerNotInstalled("No suitable RAR unpacker installed")

    assert type(params) == list, "params must be list"
    args = [rar_executable_cached] + params
    try:
        gc.disable()  # See http://bugs.python.org/issue1336
        return subprocess.Popen(args, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
    finally:
        gc.enable()