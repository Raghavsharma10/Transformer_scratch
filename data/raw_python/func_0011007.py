def run(files, temp_folder):
    """Check isort errors in the code base.

    For the --quiet option, at least isort >= 4.1.1 is required.
    https://github.com/timothycrosley/isort/blob/develop/CHANGELOG.md#411

    """
    try:
        import isort  # NOQA
    except ImportError:
        return NO_ISORT_MSG

    py_files = filter_python_files(files)

    # --quiet because isort >= 4.1 outputs its logo in the console by default.
    return bash('isort -df --quiet {0}'.format(' '.join(py_files))).value()