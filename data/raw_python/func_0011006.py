def run(files, temp_folder):
    "Check frosted errors in the code base."
    try:
        import frosted  # NOQA
    except ImportError:
        return NO_FROSTED_MSG

    py_files = filter_python_files(files)
    cmd = 'frosted {0}'.format(' '.join(py_files))

    return bash(cmd).value()