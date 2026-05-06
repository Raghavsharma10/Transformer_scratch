def list_json_files(directory, recursive=False):
    """Return a list of file paths for JSON files within `directory`.

    Args:
        directory: A path to a directory.
        recursive: If ``True``, this function will descend into all
            subdirectories.

    Returns:
        A list of JSON file paths directly under `directory`.

    """
    json_files = []

    for top, dirs, files in os.walk(directory):
        dirs.sort()
        # Get paths to each file in `files`
        paths = (os.path.join(top, f) for f in sorted(files))

        # Add all the .json files to our return collection
        json_files.extend(x for x in paths if is_json(x))

        if not recursive:
            break

    return json_files