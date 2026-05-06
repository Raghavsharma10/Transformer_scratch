def files_to_check(commit_only):
    """
    Validate the commit diff.

    Make copies of the staged changes for analysis.
    """
    global TEMP_FOLDER
    safe_directory = tempfile.mkdtemp()
    TEMP_FOLDER = safe_directory

    files = get_files(commit_only=commit_only, copy_dest=safe_directory)

    try:
        yield files
    finally:
        shutil.rmtree(safe_directory)