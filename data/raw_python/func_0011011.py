def run(files, temp_folder):
    "Check flake8 errors in the code base."
    try:
        import flake8  # NOQA
    except ImportError:
        return NO_FLAKE_MSG
    try:
        from flake8.engine import get_style_guide
    except ImportError:
        # We're on a new version of flake8
        from flake8.api.legacy import get_style_guide

    py_files = filter_python_files(files)
    if not py_files:
        return
    DEFAULT_CONFIG = join(temp_folder, get_config_file())

    with change_folder(temp_folder):
        flake8_style = get_style_guide(config_file=DEFAULT_CONFIG)
        out, err = StringIO(), StringIO()
        with redirected(out, err):
            flake8_style.check_files(py_files)
    return out.getvalue().strip() + err.getvalue().strip()