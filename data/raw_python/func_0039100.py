def working_area(files, name=""):
    """
    Copy all files to a temporary directory (the working area)
    Optionally names the working area name
    Returns path to the working area
    """
    with tempfile.TemporaryDirectory() as dir:
        dir = Path(Path(dir) / name)
        dir.mkdir(exist_ok=True)

        for f in files:
            dest = (dir / f).absolute()
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(f, dest)
        yield dir