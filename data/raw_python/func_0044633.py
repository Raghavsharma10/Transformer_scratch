def get_vswhere_path():
    """
    Get the path to vshwere.exe.

    If vswhere is not already installed as part of Visual Studio, and no
    alternate path is given using `set_vswhere_path()`, the latest release will
    be downloaded and stored alongside this script.
    """
    if alternate_path and os.path.exists(alternate_path):
        return alternate_path

    if DEFAULT_PATH and os.path.exists(DEFAULT_PATH):
        return DEFAULT_PATH

    if os.path.exists(DOWNLOAD_PATH):
        return DOWNLOAD_PATH

    _download_vswhere()
    return DOWNLOAD_PATH