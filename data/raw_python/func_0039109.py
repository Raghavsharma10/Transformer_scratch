def _glob(pattern, skip_dirs=False):
    """Glob pattern, expand directories, return all files that matched."""
    # Implicit recursive iff no / in pattern and starts with *
    if "/" not in pattern and pattern.startswith("*"):
        files = glob.glob(f"**/{pattern}", recursive=True)
    else:
        files = glob.glob(pattern, recursive=True)

    # Expand dirs
    all_files = set()
    for file in files:
        if os.path.isdir(file) and not skip_dirs:
            all_files.update(set(f for f in _glob(f"{file}/**/*", skip_dirs=True) if not os.path.isdir(f)))
        else:
            all_files.add(file)

    # Normalize all files
    return {str(Path(f)) for f in all_files}