def get_py_files(dir_name: str) -> list:
    """Get all .py files."""
    return flatten([
        x for x in
        [["{0}/{1}".format(path, f) for f in files if f.endswith(".py")]
         for path, _, files in os.walk(dir_name)
         if not path.startswith("./build")] if x
    ])