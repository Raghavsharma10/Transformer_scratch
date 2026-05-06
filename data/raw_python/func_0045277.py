def get_entries(path):
    """Return sorted lists of directories and files in the given path."""
    dirs, files = [], []
    for entry in os.listdir(path):
        # Categorize entry as directory or file.
        if os.path.isdir(os.path.join(path, entry)):
            dirs.append(entry)
        else:
            files.append(entry)
    dirs.sort()
    files.sort()
    return dirs, files